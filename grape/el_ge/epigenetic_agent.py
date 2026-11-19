# -*- coding: utf-8 -*-
"""
Bullet 1: LLM-Powered Epigenetic BNF Grammar Agent (Supports Local Ollama & Cloud APIs).

Every K generations:
1. Collects the current .bnf grammar, the top Accuracy Specialists, and the top
   Parsimony Specialists (smallest low-error formulas/programs) from the population.
2. Prompts the LLM Agent (via local Ollama e.g. `gemma4:e4b` or OpenAI/Gemini API)
   to analyze the winning expressions and propose new BNF production rules ("combos").
3. Validates that the LLM's proposed rules use only legal functions/non-terminals from
   the grammar, tests them via the Counterfactual Graft Gate, writes the new .bnf file,
   and runs Mechanic A (`remap_population_genomes`) so zero existing genomes break.
"""

import copy
import json
import math
import os
import random
import re
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .remapper import DerivationNode, build_derivation_tree, remap_population_genomes, tree_to_genome


def serialize_grammar_to_bnf(
    non_terminals: List[str],
    production_rules_by_nt: Dict[str, List[str]],
    output_path: Path,
) -> Path:
    """Write a valid GRAPE-compatible .bnf file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for nt in non_terminals:
        prods = production_rules_by_nt[nt]
        rhs = " | ".join(prods)
        lines.append(f"{nt} ::= {rhs}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _extract_lifted_template(node: DerivationNode) -> Optional[str]:
    """Fallback AST template extractor if LLM server is offline."""
    if not node.children:
        return None

    def _render_depth_limited(curr: DerivationNode, depth_left: int) -> str:
        if curr.rule_str == "<c><c>.<c><c>":
            return curr.render_phenotype()
        if depth_left <= 0:
            return curr.nt_name
        if not curr.children:
            if "x[" in curr.rule_str:
                return curr.nt_name
            return curr.rule_str
        next_depth = (
            depth_left
            if (len(curr.children) == 1 and curr.rule_str.strip() == curr.children[0].nt_name)
            else depth_left - 1
        )
        out = curr.rule_str
        for child in curr.children:
            out = out.replace(child.nt_name, _render_depth_limited(child, next_depth), 1)
        return out

    lifted = _render_depth_limited(node, depth_left=2)
    if lifted == node.rule_str or lifted == node.nt_name:
        return None
    return lifted


class EpigeneticBNFAgent:
    """
    LLM-Driven Epigenetic Architect Agent that sits above the integer genome and
    rewrites the .bnf grammar every `epoch_freq` generations.

    Supported backends:
      - `"ollama"` : Local Ollama server (`http://localhost:11434/api/generate`), e.g. `gemma4:e4b`
      - `"openai"` : OpenAI-compatible Chat Completions endpoint (`OPENAI_API_KEY`)
      - `"gemini"` : Google Gemini REST endpoint (`GEMINI_API_KEY`)
    """

    def __init__(
        self,
        initial_bnf_path: str,
        grammar_class,
        reMap_fn,
        output_dir: str = "results/grammars",
        max_rules_per_epoch: int = 1,
        max_total_added_rules: int = 3,
        elite_fraction: float = 0.12,
        backend: str = "ollama",
        model_name: str = "gemma4:e4b",
        api_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434/api/generate",
        temperature: float = 0.2,
        timeout_sec: int = 30,
        target_non_terminals: Optional[List[str]] = None,
    ):
        self.initial_bnf_path = Path(initial_bnf_path)
        self.grammar_class = grammar_class
        self.reMap_fn = reMap_fn
        self.output_dir = Path(output_dir)
        self.max_rules_per_epoch = max_rules_per_epoch
        self.max_total_added_rules = max_total_added_rules
        self.elite_fraction = elite_fraction
        self.user_target_nts = target_non_terminals

        self.backend = backend.lower().strip()
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.timeout_sec = timeout_sec

        self.added_rules_history: List[Tuple[int, str, str]] = []
        self.agent_reasoning_log: List[Dict[str, str]] = []

    def _get_current_rules_dict(self, grammar) -> Dict[str, List[str]]:
        return {
            nt: [pr[0] for pr in grammar.production_rules[idx]]
            for idx, nt in enumerate(grammar.non_terminals)
        }

    def _discover_target_non_terminals(self, grammar) -> List[str]:
        """
        Grammar- and framework-agnostic discovery of expandable compositional non-terminals.
        1. If the user explicitly passed `target_non_terminals`, use those.
        2. Otherwise, inspect the BNF production graph:
           - Preserve classic `<op>` / `<e>` preference for standard parity/regression grammars.
           - Skip unary wrapper non-terminals (k == 1, like `<Program> ::= <Init> <Body> <Measure>`)
             and pure terminal leaf tables (0 child non-terminals in all productions).
           - Select recursive multi-choice non-terminals (k >= 2) and their multi-choice
             compositional children.
        """
        if self.user_target_nts:
            valid_user = [nt for nt in self.user_target_nts if nt in grammar.non_terminals]
            if valid_user:
                return valid_user

        if "<op>" in grammar.non_terminals and "<e>" in grammar.non_terminals:
            e_idx = grammar.non_terminals.index("<e>")
            e_prods = [pr[0].strip() for pr in grammar.production_rules[e_idx]]
            if "<op>" in e_prods and len(e_prods) == 2:
                return ["<op>"]

        # If root non-terminal itself is multi-choice and recursive (e.g. `<e>` or `<log_op>`), return it
        root_nt = grammar.non_terminals[0]
        root_prods = [pr[0] for pr in grammar.production_rules[0]]
        if len(root_prods) >= 2 and any(root_nt in p for p in root_prods):
            return [root_nt]

        # General BNF topology introspection for multi-stage / statement-block grammars
        recursive_nts: List[str] = []
        compositional_nts: List[str] = []
        for idx, nt in enumerate(grammar.non_terminals):
            prods = [pr[0] for pr in grammar.production_rules[idx]]
            if len(prods) < 2:
                continue
            child_refs = [re.findall(r"\<([\(\)\w,-.]+)\>", p) for p in prods]
            total_child_refs = sum(len(refs) for refs in child_refs)
            if total_child_refs == 0:
                # Pure terminal leaf list (e.g. qubits, constants, variable names)
                continue
            is_self_recursive = any(nt in p for p in prods)
            if is_self_recursive:
                recursive_nts.append(nt)
            else:
                compositional_nts.append(nt)

        if recursive_nts:
            # Prefer deeper recursive statement/block lists over top-level outer loop wrappers
            discovered = list(reversed(recursive_nts))[:2]
            for c_nt in compositional_nts:
                if len(discovered) < 3:
                    discovered.append(c_nt)
            return discovered
        if compositional_nts:
            return compositional_nts[:2]
        return [root_nt]

    def _target_non_terminal(self, grammar) -> str:
        """Select the primary recursive expression/operator/statement non-terminal to evolve."""
        return self._discover_target_non_terminals(grammar)[0]

    def _max_rule_len(self, grammar) -> int:
        """Compute an adaptive maximum rule length from the initial grammar's longest production."""
        max_existing = max(
            (len(pr[0]) for idx in range(len(grammar.non_terminals)) for pr in grammar.production_rules[idx]),
            default=40,
        )
        return max(65, min(160, int(max_existing * 1.5)))

    def _build_agent_prompt(
        self,
        generation: int,
        grammar,
        target_nt: str,
        acc_elites: list,
        parsimony_elites: list,
        ast_suggestions: List[Tuple[str, str, float]],
    ) -> str:
        rules_dict = self._get_current_rules_dict(grammar)
        bnf_text = "\n".join(f"{nt} ::= {' | '.join(prods)}" for nt, prods in rules_dict.items())

        acc_lines = [
            f"  {i+1}. fitness={ind.fitness.values[0]:.5f} | nodes={ind.nodes} | expr: {ind.phenotype}"
            for i, ind in enumerate(acc_elites[:5])
        ]
        pars_lines = [
            f"  {i+1}. fitness={ind.fitness.values[0]:.5f} | nodes={ind.nodes} | expr: {ind.phenotype}"
            for i, ind in enumerate(parsimony_elites[:3])
        ]
        sub_hints = [f"  - {r}" for _nt, r, _s in ast_suggestions[:5]]
        t_idx = grammar.non_terminals.index(target_nt)
        sample_prods = [pr[0].strip() for pr in grammar.production_rules[t_idx][:2]]
        example_str = " or ".join(f"`{p}`" for p in sample_prods) if sample_prods else f"`{target_nt}` expansion"
        max_chars = self._max_rule_len(grammar)

        return (
            "You are the Epigenetic BNF Grammar Architect for Grammatical Evolution (GRAPE).\n"
            f"At Generation {generation}, inspect the current BNF grammar and the winning phenotypes.\n"
            "Your task is to propose 1 to 3 compact, syntactically valid production rules to append to "
            f"the non-terminal `{target_nt}` so future generations can assemble high-fitness expressions faster "
            "while preserving expressivity for unconventional solutions.\n\n"
            f"=== CURRENT BNF GRAMMAR ===\n{bnf_text}\n\n"
            "=== TOP ACCURACY ELITES (Lowest Error) ===\n"
            + "\n".join(acc_lines)
            + "\n\n=== TOP PARSIMONY SPECIALISTS (Smallest Low-Error Trees) ===\n"
            + "\n".join(pars_lines)
            + "\n\n=== RECURRING SUBTREE CANDIDATES FROM AST ===\n"
            + ("\n".join(sub_hints) if sub_hints else "  (none)")
            + "\n\nSTRICT CONSTRAINTS:\n"
            f"1. Every proposed rule MUST be a valid right-hand-side expansion for `{target_nt}` using ONLY "
            "the exact function names, operators, variables, and non-terminals (`<...>`) already present in the BNF grammar above.\n"
            f"2. Keep rules short (<= {max_chars} chars) and contain at most 2 non-terminal `<...>` slots so tree depth does not explode.\n"
            f"3. Do NOT repeat existing single rules; propose composite right-hand-side expressions (WITHOUT `{target_nt} ::=`), e.g. {example_str}.\n"
            "4. Respond ONLY with a valid JSON object in this exact format:\n"
            '{"reasoning": "1-sentence explanation", "proposed_rules": ["rhs_expr_1", "rhs_expr_2"]}\n'
        )

    def _call_llm(self, prompt: str) -> Optional[Dict]:
        """Call Ollama, OpenAI, or Gemini and parse JSON response."""
        try:
            if self.backend == "ollama":
                payload = json.dumps({
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "format": "json",
                    "options": {"temperature": self.temperature, "num_predict": 256},
                }).encode("utf-8")
                req = urllib.request.Request(
                    self.ollama_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    raw_text = body.get("response", "").strip()
                    return json.loads(raw_text)

            if self.backend == "openai":
                if not self.api_key:
                    return None
                url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
                payload = json.dumps({
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "response_format": {"type": "json_object"},
                }).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    content = body["choices"][0]["message"]["content"]
                    return json.loads(content)

            if self.backend == "gemini":
                if not self.api_key:
                    return None
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{self.model_name}:generateContent?key={self.api_key}"
                )
                payload = json.dumps({
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": self.temperature,
                        "responseMimeType": "application/json",
                    },
                }).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    content = body["candidates"][0]["content"]["parts"][0]["text"]
                    return json.loads(content)
        except Exception:
            return None
        return None

    def _is_syntactically_safe_rule(self, rule_str: str, grammar) -> bool:
        """Verify that `rule_str` only references valid non-terminals and has balanced parens."""
        max_len = self._max_rule_len(grammar)
        if not rule_str or "|" in rule_str or "::=" in rule_str or len(rule_str) > max_len:
            return False
        if rule_str.count("(") != rule_str.count(")"):
            return False
        if rule_str.count("[") != rule_str.count("]"):
            return False
        if rule_str.count("<") != rule_str.count(">"):
            return False
        referenced_nts = ["<" + m + ">" for m in re.findall(r"\<([\(\)\w,-.]+)\>", rule_str)]
        if len(referenced_nts) > 3:
            return False
        for nt in referenced_nts:
            if nt not in grammar.non_terminals:
                return False
        return True

    def mine_elite_motifs(
        self,
        population: list,
        grammar,
        max_tree_depth: int = 35,
        codon_consumption: str = "lazy",
        evaluate_fn=None,
        points_train=None,
    ) -> Tuple[List[Tuple[str, str, float]], list, list]:
        """
        Mine candidate BNF rules from Accuracy Specialists + Parsimony Specialists.
        """
        valid = [
            ind for ind in population
            if not getattr(ind, "invalid", True)
            and ind.fitness.valid
            and not math.isnan(ind.fitness.values[0])
            and getattr(ind, "used_codons", 0) >= 2
        ]
        if not valid:
            return [], [], []

        minimize = valid[0].fitness.weights[0] < 0
        sorted_by_acc = sorted(valid, key=lambda x: x.fitness.values[0], reverse=not minimize)
        n_top_acc = max(3, min(15, int(len(valid) * self.elite_fraction)))
        acc_elites = sorted_by_acc[:n_top_acc]

        top_quartile = sorted_by_acc[: max(4, len(valid) // 4)]
        parsimony_elites = sorted(top_quartile, key=lambda x: (x.nodes, x.used_codons))[: max(2, n_top_acc // 2)]

        seen_ids = set()
        elites = []
        for ind in acc_elites + parsimony_elites:
            if id(ind) not in seen_ids:
                seen_ids.add(id(ind))
                elites.append(ind)

        existing_rules: Dict[str, Set[str]] = {
            nt: {pr[0].replace(" ", "") for pr in grammar.production_rules[i]}
            for i, nt in enumerate(grammar.non_terminals)
        }

        target_nts = self._discover_target_non_terminals(grammar)
        max_lifted_len = self._max_rule_len(grammar)
        is_multistage = len(target_nts) > 1 or target_nts[0] != grammar.non_terminals[0]

        motif_elite_hits: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
        motif_rank_weight: Dict[Tuple[str, str], float] = defaultdict(float)

        for rank_idx, ind in enumerate(elites):
            root, _, _ = build_derivation_tree(
                ind.genome, grammar, max_depth=max_tree_depth, codon_consumption=codon_consumption
            )
            if root is None:
                continue
            weight = 1.0 / (1.0 + 0.25 * rank_idx)

            def _walk(node: DerivationNode) -> None:
                for child in node.children:
                    _walk(child)

                if node.nt_name not in target_nts:
                    return
                sub_depth = node.max_subtree_depth()
                if sub_depth < 2 or sub_depth > (6 if is_multistage else 4):
                    return

                if 2 <= sub_depth <= (5 if is_multistage else 3) and len(node.children) >= 1:
                    concrete = node.render_phenotype()
                    if ("x[" in concrete) and len(concrete) <= 50 and node.count_terminal_nodes() >= 2:
                        norm_c = concrete.replace(" ", "")
                        if norm_c not in existing_rules[node.nt_name]:
                            if any(op in concrete for op in ("+", "-", "*")) and not concrete.startswith("("):
                                concrete = f"({concrete})"
                            key_c = (node.nt_name, concrete)
                            motif_elite_hits[key_c].add(rank_idx)
                            motif_rank_weight[key_c] += weight * 1.5

                lifted = _extract_lifted_template(node)
                if lifted and len(lifted) <= max_lifted_len and "<" in lifted and ">" in lifted:
                    arity = len(re.findall(r"\<([\(\)\w,-.]+)\>", lifted))
                    norm = lifted.replace(" ", "")
                    max_arity = 3 if is_multistage else 2
                    if 1 <= arity <= max_arity and norm not in existing_rules[node.nt_name] and norm != node.nt_name:
                        key = (node.nt_name, lifted)
                        motif_elite_hits[key].add(rank_idx)
                        motif_rank_weight[key] += weight * (1.4 if arity == 1 else 1.1)

            _walk(root)

        scored_candidates: List[Tuple[str, str, float]] = []
        fallback_candidates: List[Tuple[str, str, float]] = []
        for (nt_name, rule_str), hit_set in motif_elite_hits.items():
            score = motif_rank_weight[(nt_name, rule_str)] * math.log2(1.0 + len(rule_str) / 6.0)
            if len(hit_set) >= 2:
                scored_candidates.append((nt_name, rule_str, score))
            else:
                fallback_candidates.append((nt_name, rule_str, score))

        scored_candidates.sort(key=lambda item: item[2], reverse=True)
        if not scored_candidates:
            fallback_candidates.sort(key=lambda item: item[2], reverse=True)
            return fallback_candidates, acc_elites, parsimony_elites
        return scored_candidates, acc_elites, parsimony_elites

    def _verify_and_graft_rule(
        self,
        nt_name: str,
        rule_str: str,
        test_grammar,
        elites: list,
        evaluate_fn,
        points_train,
        max_tree_depth: int,
        codon_size: int,
        codon_consumption: str,
        rng: random.Random,
    ) -> List[Tuple[List[int], float]]:
        """
        Test grafting `rule_str` into leaf/deep nodes of top elites on `points_train`.
        """
        if evaluate_fn is None or points_train is None or not elites:
            return []

        nt_idx = test_grammar.non_terminals.index(nt_name)
        prods = [pr[0] for pr in test_grammar.production_rules[nt_idx]]
        if rule_str not in prods:
            return []
        new_rule_idx = prods.index(rule_str)
        k_new = test_grammar.n_rules[nt_idx]
        child_nts = ["<" + m + ">" for m in re.findall(r"\<([\(\)\w,-.]+)\>", rule_str)]

        minimize = elites[0].fitness.weights[0] < 0
        winning_grafts: List[Tuple[List[int], float]] = []

        probe_ind = copy.deepcopy(elites[0])
        for elite in elites[:4]:
            root, _, tail = build_derivation_tree(
                elite.genome, test_grammar, max_depth=max_tree_depth, codon_consumption=codon_consumption
            )
            if root is None:
                continue

            matching_nodes: List[DerivationNode] = []

            def _collect(curr: DerivationNode) -> None:
                for c in curr.children:
                    _collect(c)
                if curr.nt_name == nt_name:
                    matching_nodes.append(curr)

            _collect(root)
            if not matching_nodes:
                continue

            for target_node in matching_nodes[:6]:
                old_r_idx = target_node.rule_index
                old_r_str = target_node.rule_str
                old_codon = target_node.codon_value
                old_k = target_node.k_choices
                old_children = list(target_node.children)

                existing_same_nt = [c for c in old_children if c.nt_name in child_nts]
                new_children: List[DerivationNode] = []
                valid_graft = True
                for req_nt in child_nts:
                    match_c = next((c.clone() for c in existing_same_nt if c.nt_name == req_nt), None)
                    if match_c is not None:
                        new_children.append(match_c)
                    elif req_nt == nt_name:
                        wrapped = DerivationNode(
                            nt_name=nt_name,
                            nt_index=nt_idx,
                            rule_index=old_r_idx,
                            rule_str=old_r_str,
                            codon_value=old_codon,
                            k_choices=old_k,
                            children=old_children,
                        )
                        new_children.append(wrapped)
                    else:
                        valid_graft = False
                        break

                if not valid_graft:
                    continue

                target_node.rule_index = new_rule_idx
                target_node.rule_str = rule_str
                target_node.codon_value = new_rule_idx
                target_node.k_choices = k_new
                target_node.children = new_children

                if root.max_subtree_depth() <= max_tree_depth:
                    cand_genome = tree_to_genome(
                        root,
                        test_grammar,
                        codon_size=codon_size,
                        codon_consumption=codon_consumption,
                        tail=tail,
                        rng=rng,
                    )
                    self.reMap_fn(probe_ind, cand_genome, test_grammar, max_tree_depth, codon_consumption)
                    if not probe_ind.invalid:
                        cand_fit = evaluate_fn(probe_ind, points_train)[0]
                        if not math.isnan(cand_fit):
                            elite_fit = elite.fitness.values[0]
                            beats_or_ties_smaller = (
                                (cand_fit < elite_fit - 1e-9)
                                if minimize
                                else (cand_fit > elite_fit + 1e-9)
                            ) or (
                                abs(cand_fit - elite_fit) <= 1e-9
                                and probe_ind.used_codons < elite.used_codons
                            )
                            if beats_or_ties_smaller:
                                winning_grafts.append((cand_genome, float(cand_fit)))

                target_node.rule_index = old_r_idx
                target_node.rule_str = old_r_str
                target_node.codon_value = old_codon
                target_node.k_choices = old_k
                target_node.children = old_children

        winning_grafts.sort(key=lambda x: x[1], reverse=not minimize)
        return winning_grafts

    def step_epoch(
        self,
        generation: int,
        population: list,
        current_grammar,
        problem_tag: str,
        seed: int,
        max_tree_depth: int = 35,
        codon_size: int = 255,
        codon_consumption: str = "lazy",
        evaluate_fn=None,
        points_train=None,
    ) -> Tuple[object, int]:
        """
        Query the LLM Agent (Ollama / OpenAI / Gemini) for proposed .bnf rules,
        validate & graft-test them, update the .bnf file, and remap population codons.
        """
        remaining_budget = self.max_total_added_rules - len(self.added_rules_history)
        if remaining_budget <= 0:
            return current_grammar, 0

        saved_rng_state = random.getstate()
        local_rng = random.Random(31337 + seed * 100 + generation)
        try:
            ast_candidates, acc_elites, parsimony_elites = self.mine_elite_motifs(
                population,
                current_grammar,
                max_tree_depth=max_tree_depth,
                codon_consumption=codon_consumption,
                evaluate_fn=evaluate_fn,
                points_train=points_train,
            )
            if not acc_elites:
                return current_grammar, 0

            target_nt = self._target_non_terminal(current_grammar)

            # 1. Select source of new BNF rules: Pure LLM vs. Pure Deterministic AST
            candidate_queue: List[Tuple[str, str, float]] = []
            if self.backend in ("ollama", "openai", "gemini"):
                prompt = self._build_agent_prompt(
                    generation, current_grammar, target_nt, acc_elites, parsimony_elites, []
                )
                llm_resp = self._call_llm(prompt)
                if isinstance(llm_resp, dict):
                    reasoning = str(llm_resp.get("reasoning", ""))
                    proposed = llm_resp.get("proposed_rules", [])
                    if isinstance(proposed, list):
                        for r_str in proposed:
                            if not isinstance(r_str, str):
                                continue
                            clean_r = r_str.strip()
                            if "::=" in clean_r:
                                clean_r = clean_r.split("::=", 1)[1].strip()
                            # Split if LLM combined multiple choices with '|'
                            for part in clean_r.split("|"):
                                part = part.strip()
                                if self._is_syntactically_safe_rule(part, current_grammar):
                                    candidate_queue.append((target_nt, part, 100.0))
                        self.agent_reasoning_log.append({
                            "generation": str(generation),
                            "model": f"{self.backend}:{self.model_name}",
                            "reasoning": reasoning,
                            "proposed": str([c[1] for c in candidate_queue]),
                        })
            else:
                # Pure deterministic AST mining (`EL_GE_AST`)
                candidate_queue.extend(ast_candidates)

            if not candidate_queue:
                return current_grammar, 0

            valid = [
                ind for ind in population
                if not getattr(ind, "invalid", True)
                and ind.fitness.valid
                and not math.isnan(ind.fitness.values[0])
            ]
            minimize = valid[0].fitness.weights[0] < 0 if valid else True
            sorted_valid = sorted(valid, key=lambda x: x.fitness.values[0], reverse=not minimize)
            elites = sorted_valid[:6]

            n_to_add = min(self.max_rules_per_epoch, remaining_budget)
            committed_rules: List[Tuple[str, str]] = []
            committed_grammar = current_grammar
            best_grafts_to_inject: List[Tuple[List[int], float]] = []

            for nt_name, rule_str, _score in candidate_queue[:10]:
                if len(committed_rules) >= n_to_add:
                    break
                rules_dict = self._get_current_rules_dict(committed_grammar)
                existing_norm = {r.replace(" ", "") for r in rules_dict[nt_name]}
                if rule_str.replace(" ", "") in existing_norm:
                    continue
                if len(rules_dict[nt_name]) + 1 > codon_size:
                    continue

                rules_dict[nt_name].append(rule_str)
                bnf_out_path = self.output_dir / f"{problem_tag}_seed{seed}_gen{generation:03d}.bnf"
                serialize_grammar_to_bnf(committed_grammar.non_terminals, rules_dict, bnf_out_path)

                try:
                    trial_grammar = self.grammar_class(str(bnf_out_path))
                except Exception:
                    continue

                # Verify the rule is executable when grafted and collect any superior grafts
                if evaluate_fn is not None and points_train is not None and elites:
                    grafts = self._verify_and_graft_rule(
                        nt_name,
                        rule_str,
                        trial_grammar,
                        elites,
                        evaluate_fn=evaluate_fn,
                        points_train=points_train,
                        max_tree_depth=max_tree_depth,
                        codon_size=codon_size,
                        codon_consumption=codon_consumption,
                        rng=local_rng,
                    )
                    if grafts:
                        best_grafts_to_inject.extend(grafts[:2])

                committed_grammar = trial_grammar
                committed_rules.append((nt_name, rule_str))

            if not committed_rules:
                return current_grammar, 0

            # Mechanic A: Adjust all genomes in population so phenotypes are 100% preserved
            remap_population_genomes(
                population,
                old_grammar=current_grammar,
                new_grammar=committed_grammar,
                reMap_fn=self.reMap_fn,
                max_tree_depth=max_tree_depth,
                codon_size=codon_size,
                codon_consumption=codon_consumption,
            )

            if best_grafts_to_inject and len(sorted_valid) > len(best_grafts_to_inject) + 2:
                worst_inds = sorted_valid[-len(best_grafts_to_inject) :]
                for target_ind, (g_genome, g_fit) in zip(worst_inds, best_grafts_to_inject):
                    self.reMap_fn(target_ind, g_genome, committed_grammar, max_tree_depth, codon_consumption)
                    if evaluate_fn is not None and points_train is not None:
                        target_ind.fitness.values = evaluate_fn(target_ind, points_train)
                    else:
                        target_ind.fitness.values = (g_fit,)

            for nt_name, rule_str in committed_rules:
                self.added_rules_history.append((generation, nt_name, rule_str))

            return committed_grammar, len(committed_rules)
        finally:
            random.setstate(saved_rng_state)
