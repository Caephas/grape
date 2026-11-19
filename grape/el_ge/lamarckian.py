# -*- coding: utf-8 -*-
"""
Mechanic B: Lamarckian Derivation-Tree Simplification, Causal Knock-Out Pruning,
Constant Polishing, and Codon Writeback.

Guarantees monotonic fitness improvement/preservation on elites:
  Fitness(after_lamarckian) <= Fitness(before_lamarckian) (for minimization)
and uses an isolated RNG so the baseline evolutionary RNG stream is never perturbed.
"""

import math
import random
from typing import List, Optional, Tuple

from .remapper import DerivationNode, build_derivation_tree, tree_to_genome


def _make_digit_constant_node(val: float, nt_name: str, grammar) -> Optional[DerivationNode]:
    """Construct a `<c><c>.<c><c>` DerivationNode for `val` in [00.00, 99.99]."""
    if nt_name not in grammar.non_terminals or "<c>" not in grammar.non_terminals:
        return None
    nt_idx = grammar.non_terminals.index(nt_name)
    c_idx = grammar.non_terminals.index("<c>")
    prod_strs = [pr[0] for pr in grammar.production_rules[nt_idx]]
    if "<c><c>.<c><c>" not in prod_strs:
        return None
    rule_idx = prod_strs.index("<c><c>.<c><c>")
    c_prods = [pr[0] for pr in grammar.production_rules[c_idx]]

    clamped = max(0.0, min(99.99, abs(float(val))))
    formatted = f"{clamped:05.2f}"
    digits = [formatted[0], formatted[1], formatted[3], formatted[4]]
    children: List[DerivationNode] = []
    for d in digits:
        if d not in c_prods:
            return None
        d_idx = c_prods.index(d)
        children.append(
            DerivationNode(
                nt_name="<c>",
                nt_index=c_idx,
                rule_index=d_idx,
                rule_str=d,
                codon_value=d_idx,
                k_choices=grammar.n_rules[c_idx],
                children=[],
                phenotype_cache=d,
            )
        )
    return DerivationNode(
        nt_name=nt_name,
        nt_index=nt_idx,
        rule_index=rule_idx,
        rule_str="<c><c>.<c><c>",
        codon_value=rule_idx,
        k_choices=grammar.n_rules[nt_idx],
        children=children,
        phenotype_cache=formatted,
    )


def _is_zero_phenotype(pheno: str) -> bool:
    try:
        return "x[" not in pheno and abs(float(pheno)) < 1e-9
    except ValueError:
        return False


def _is_one_phenotype(pheno: str) -> bool:
    try:
        return "x[" not in pheno and abs(float(pheno) - 1.0) < 1e-9
    except ValueError:
        return False


def simplify_derivation_tree(node: DerivationNode, grammar) -> Tuple[DerivationNode, bool]:
    """
    Bottom-up exact algebraic and boolean simplification of a BNF derivation tree.
    """
    changed = False
    new_children: List[DerivationNode] = []
    for child in node.children:
        s_child, c_changed = simplify_derivation_tree(child, grammar)
        new_children.append(s_child)
        if c_changed:
            changed = True
    node.children = new_children
    node.phenotype_cache = node.render_phenotype()

    rule = node.rule_str.replace(" ", "")

    # 1. Wrapper pass-through rules like `<e> ::= <op> | <x>` in parity4.bnf
    if len(node.children) == 1 and node.children[0].nt_name == "<op>":
        op_node = node.children[0]
        op_rule = op_node.rule_str.replace(" ", "")
        if len(op_node.children) == 2:
            left, right = op_node.children[0], op_node.children[1]
            l_str = left.render_phenotype()
            r_str = right.render_phenotype()
            if op_rule in ("and_(<e>,<e>)", "or_(<e>,<e>)") and l_str == r_str and left.nt_name == node.nt_name:
                return left, True
            if op_rule == "nand_(<e>,<e>)" and l_str == r_str and len(left.children) == 1:
                inner_op = left.children[0]
                if inner_op.nt_name == "<op>" and inner_op.rule_str.replace(" ", "") == "nand_(<e>,<e>)":
                    op_idx = grammar.non_terminals.index("<op>")
                    prods = [pr[0].replace(" ", "") for pr in grammar.production_rules[op_idx]]
                    if "and_(<e>,<e>)" in prods:
                        and_idx = prods.index("and_(<e>,<e>)")
                        inner_op.rule_index = and_idx
                        inner_op.rule_str = grammar.production_rules[op_idx][and_idx][0]
                        inner_op.codon_value = and_idx
                        left.phenotype_cache = left.render_phenotype()
                        return left, True
            if op_rule == "nor_(<e>,<e>)" and l_str == r_str and len(left.children) == 1:
                inner_op = left.children[0]
                if inner_op.nt_name == "<op>" and inner_op.rule_str.replace(" ", "") == "nor_(<e>,<e>)":
                    op_idx = grammar.non_terminals.index("<op>")
                    prods = [pr[0].replace(" ", "") for pr in grammar.production_rules[op_idx]]
                    if "or_(<e>,<e>)" in prods:
                        or_idx = prods.index("or_(<e>,<e>)")
                        inner_op.rule_index = or_idx
                        inner_op.rule_str = grammar.production_rules[op_idx][or_idx][0]
                        inner_op.codon_value = or_idx
                        left.phenotype_cache = left.render_phenotype()
                        return left, True

    # 2. Direct boolean rules (`heartDisease.bnf`)
    if rule in ("and_(<log_op>,<log_op>)", "or_(<log_op>,<log_op>)") and len(node.children) == 2:
        left, right = node.children[0], node.children[1]
        if left.render_phenotype() == right.render_phenotype() and left.nt_name == node.nt_name:
            return left, True

    if rule == "not_(<log_op>)" and len(node.children) == 1:
        child = node.children[0]
        if child.rule_str.replace(" ", "") == "not_(<log_op>)" and len(child.children) == 1:
            return child.children[0], True

    # 3. Binary arithmetic rules
    if len(node.children) == 2:
        left, right = node.children[0], node.children[1]
        l_pheno = left.render_phenotype()
        r_pheno = right.render_phenotype()

        if rule in ("<e>-<e>", "sub(<num_op>,<num_op>)"):
            if l_pheno == r_pheno:
                zero_node = _make_digit_constant_node(0.0, node.nt_name, grammar)
                if zero_node is not None:
                    return zero_node, True
            if _is_zero_phenotype(r_pheno) and left.nt_name == node.nt_name:
                return left, True
            l_rule = left.rule_str.replace(" ", "")
            if l_rule in ("<e>+<e>", "add(<num_op>,<num_op>)") and len(left.children) == 2:
                ll, lr = left.children[0], left.children[1]
                if lr.render_phenotype() == r_pheno and ll.nt_name == node.nt_name:
                    return ll, True
                if ll.render_phenotype() == r_pheno and lr.nt_name == node.nt_name:
                    return lr, True

        if rule in ("<e>+<e>", "add(<num_op>,<num_op>)"):
            if _is_zero_phenotype(r_pheno) and left.nt_name == node.nt_name:
                return left, True
            if _is_zero_phenotype(l_pheno) and right.nt_name == node.nt_name:
                return right, True
            r_rule = right.rule_str.replace(" ", "")
            if r_rule in ("<e>-<e>", "sub(<num_op>,<num_op>)") and len(right.children) == 2:
                rl, rr = right.children[0], right.children[1]
                if rl.render_phenotype() == rr.render_phenotype() and left.nt_name == node.nt_name:
                    return left, True
                if rr.render_phenotype() == l_pheno and rl.nt_name == node.nt_name:
                    return rl, True
            l_rule = left.rule_str.replace(" ", "")
            if l_rule in ("<e>-<e>", "sub(<num_op>,<num_op>)") and len(left.children) == 2:
                ll, lr = left.children[0], left.children[1]
                if ll.render_phenotype() == lr.render_phenotype() and right.nt_name == node.nt_name:
                    return right, True
                if lr.render_phenotype() == r_pheno and ll.nt_name == node.nt_name:
                    return ll, True

        if rule in ("pdiv(<e>,<e>)", "pdiv(<num_op>,<num_op>)"):
            if l_pheno == r_pheno:
                one_node = _make_digit_constant_node(1.0, node.nt_name, grammar)
                if one_node is not None:
                    return one_node, True
            if _is_one_phenotype(r_pheno) and left.nt_name == node.nt_name:
                return left, True
            l_rule = left.rule_str.replace(" ", "")
            if l_rule in ("<e>*<e>", "mul(<num_op>,<num_op>)") and len(left.children) == 2:
                ll, lr = left.children[0], left.children[1]
                if lr.render_phenotype() == r_pheno and ll.nt_name == node.nt_name:
                    return ll, True
                if ll.render_phenotype() == r_pheno and lr.nt_name == node.nt_name:
                    return lr, True

        if rule in ("<e>*<e>", "mul(<num_op>,<num_op>)"):
            if _is_one_phenotype(r_pheno) and left.nt_name == node.nt_name:
                return left, True
            if _is_one_phenotype(l_pheno) and right.nt_name == node.nt_name:
                return right, True
            if _is_zero_phenotype(r_pheno) or _is_zero_phenotype(l_pheno):
                zero_node = _make_digit_constant_node(0.0, node.nt_name, grammar)
                if zero_node is not None:
                    return zero_node, True
            r_rule = right.rule_str.replace(" ", "")
            if r_rule in ("pdiv(<e>,<e>)", "pdiv(<num_op>,<num_op>)") and len(right.children) == 2:
                rl, rr = right.children[0], right.children[1]
                if rl.render_phenotype() == rr.render_phenotype() and left.nt_name == node.nt_name:
                    return left, True
            l_rule = left.rule_str.replace(" ", "")
            if l_rule in ("pdiv(<e>,<e>)", "pdiv(<num_op>,<num_op>)") and len(left.children) == 2:
                ll, lr = left.children[0], left.children[1]
                if ll.render_phenotype() == lr.render_phenotype() and right.nt_name == node.nt_name:
                    return right, True

    # 4. Ternary `(<e> <op> <e>)` in Banknote.bnf
    if rule == "(<e><op><e>)" and len(node.children) == 3:
        left, op_node, right = node.children[0], node.children[1], node.children[2]
        op_sym = op_node.render_phenotype().strip()
        l_pheno = left.render_phenotype()
        r_pheno = right.render_phenotype()
        if op_sym in ("-", "+") and _is_zero_phenotype(r_pheno):
            return left, True
        if op_sym == "+" and _is_zero_phenotype(l_pheno):
            return right, True
        if op_sym == "*" and _is_one_phenotype(r_pheno):
            return left, True
        if op_sym == "*" and _is_one_phenotype(l_pheno):
            return right, True
        if op_sym == "-" and left.rule_str.replace(" ", "") == "(<e><op><e>)" and len(left.children) == 3:
            if left.children[1].render_phenotype().strip() == "+":
                if left.children[2].render_phenotype() == r_pheno:
                    return left.children[0], True
                if left.children[0].render_phenotype() == r_pheno:
                    return left.children[2], True

    return node, changed


def _causal_knockout_prune(
    root: DerivationNode,
    ind,
    grammar,
    reMap_fn,
    evaluate_fn,
    points_train,
    tail: List[int],
    max_tree_depth: int,
    codon_size: int,
    codon_consumption: str,
    rng: random.Random,
    max_probes: int = 8,
) -> Tuple[DerivationNode, float, bool]:
    """
    Causal Subtree Knock-Out Pruning:
    Tests replacing internal wrapper/operator nodes with one of their same-typed
    child subtrees. If the smaller tree achieves strictly better or equal training
    fitness (`cand_fit <= best_fit`), the bloat wrapper is permanently pruned!
    """
    best_fit = ind.fitness.values[0]
    minimize = ind.fitness.weights[0] < 0
    pruned_any = False

    # Collect candidate (parent_node, child_index, replacement_subtree) triples
    candidates: List[Tuple[DerivationNode, int, DerivationNode]] = []

    def _find_replaceable(parent: DerivationNode) -> None:
        for idx, child in enumerate(parent.children):
            _find_replaceable(child)
            # Check if `child` has a direct descendant with the exact same `nt_name`
            for grandchild in child.children:
                if grandchild.nt_name == child.nt_name:
                    candidates.append((parent, idx, grandchild))
                # Or 2-step wrapper `<e> -> <op> -> and_(<e>, <e>)`
                for ggc in grandchild.children:
                    if ggc.nt_name == child.nt_name:
                        candidates.append((parent, idx, ggc))

    _find_replaceable(root)
    if not candidates:
        return root, best_fit, False

    rng.shuffle(candidates)
    for parent, idx, replacement in candidates[:max_probes]:
        original_child = parent.children[idx]
        if replacement.count_terminal_nodes() >= original_child.count_terminal_nodes():
            continue
        parent.children[idx] = replacement
        cand_genome = tree_to_genome(
            root, grammar, codon_size=codon_size, codon_consumption=codon_consumption, tail=tail, rng=rng
        )
        reMap_fn(ind, cand_genome, grammar, max_tree_depth, codon_consumption)
        if not ind.invalid:
            cand_fit = evaluate_fn(ind, points_train)[0]
            is_better_or_equal = (
                not math.isnan(cand_fit)
                and ((cand_fit <= best_fit + 1e-12) if minimize else (cand_fit >= best_fit - 1e-12))
            )
            if is_better_or_equal:
                best_fit = cand_fit
                pruned_any = True
                continue
        # Revert if knock-out hurt fitness
        parent.children[idx] = original_child

    return root, best_fit, pruned_any


def _collect_constant_nodes(node: DerivationNode, out: List[DerivationNode]) -> None:
    if node.rule_str == "<c><c>.<c><c>" and len(node.children) == 4:
        out.append(node)
        return
    if node.nt_name == "<c>" and not node.children:
        out.append(node)
        return
    for c in node.children:
        _collect_constant_nodes(c, out)


def lamarckian_simplify_individual(
    ind,
    grammar,
    reMap_fn,
    evaluate_fn=None,
    points_train=None,
    max_tree_depth: int = 35,
    codon_size: int = 255,
    codon_consumption: str = "lazy",
    tune_constants: bool = False,
    n_tune_steps: int = 8,
    rng: Optional[random.Random] = None,
    force_simplify: bool = False,
    existing_phenotypes: Optional[set] = None,
) -> bool:
    """
    Mechanic B on a single individual:
    1. Builds the BNF derivation tree.
    2. Runs bottom-up algebraic/boolean simplification (`simplify_derivation_tree`).
    3. Runs Causal Knock-Out Pruning (`_causal_knockout_prune`) & Constant Polishing
       when `evaluate_fn` is provided.
    4. Compiles the cleaned derivation tree back into `ind.genome` ONLY if fitness is
       preserved or improved.
    """
    if getattr(ind, "invalid", True):
        return False

    _rng = rng if rng is not None else random.Random(42)
    old_genome = list(ind.genome)
    old_fitness = ind.fitness.values[0] if ind.fitness.valid else None

    root, used_codons, tail = build_derivation_tree(
        old_genome, grammar, max_depth=max_tree_depth, codon_consumption=codon_consumption
    )
    if root is None:
        return False

    min_reserve_tail = max(25, used_codons // 2)
    if len(tail) >= min_reserve_tail:
        clean_tail = tail[: max(min_reserve_tail, min(len(tail), 150))]
    else:
        clean_tail = list(tail) + [_rng.randint(0, codon_size) for _ in range(min_reserve_tail - len(tail))]

    simplified_root, tree_changed = simplify_derivation_tree(root.clone(), grammar)

    pruned_any = False
    const_improved = False
    best_fit = old_fitness
    is_bloated = force_simplify or (used_codons > 24)

    # If the individual is not bloated (used_codons <= 24), do not strip neutral
    # algebraic stepping stones unless constant tuning strictly improves fitness.
    if not is_bloated:
        simplified_root = root.clone()
        tree_changed = False

    if tune_constants and evaluate_fn is not None and points_train is not None and old_fitness is not None:
        # Step A: Causal Knock-Out Bloat Pruning (only on bloated trees: used_codons > 20)
        if is_bloated:
            simplified_root, best_fit, pruned_any = _causal_knockout_prune(
                simplified_root,
                ind,
                grammar,
                reMap_fn=reMap_fn,
                evaluate_fn=evaluate_fn,
                points_train=points_train,
                tail=clean_tail,
                max_tree_depth=max_tree_depth,
                codon_size=codon_size,
                codon_consumption=codon_consumption,
                rng=_rng,
            )

        # Step B: Coordinate Constant Tuning on `<c>` leaves
        const_nodes: List[DerivationNode] = []
        _collect_constant_nodes(simplified_root, const_nodes)
        if const_nodes and "<c>" in grammar.non_terminals:
            c_idx = grammar.non_terminals.index("<c>")
            k_c = grammar.n_rules[c_idx]
            minimize = ind.fitness.weights[0] < 0

            for _ in range(n_tune_steps):
                target = _rng.choice(const_nodes)
                digit_child = _rng.choice(target.children) if (target.rule_str == "<c><c>.<c><c>" and len(target.children) == 4) else target

                old_rule_idx = digit_child.rule_index
                old_rule_str = digit_child.rule_str
                old_codon = digit_child.codon_value

                delta = _rng.choice([-1, 1])
                new_rule_idx = (old_rule_idx + delta) % k_c
                digit_child.rule_index = new_rule_idx
                digit_child.rule_str = grammar.production_rules[c_idx][new_rule_idx][0]
                digit_child.codon_value = new_rule_idx

                cand_genome = tree_to_genome(
                    simplified_root,
                    grammar,
                    codon_size=codon_size,
                    codon_consumption=codon_consumption,
                    tail=clean_tail,
                    rng=_rng,
                )
                reMap_fn(ind, cand_genome, grammar, max_tree_depth, codon_consumption)
                if not ind.invalid:
                    cand_fit = evaluate_fn(ind, points_train)[0]
                    is_better = (
                        not math.isnan(cand_fit)
                        and ((cand_fit < best_fit - 1e-9) if minimize else (cand_fit > best_fit + 1e-9))
                    )
                    if is_better:
                        best_fit = cand_fit
                        const_improved = True
                        continue

                digit_child.rule_index = old_rule_idx
                digit_child.rule_str = old_rule_str
                digit_child.codon_value = old_codon

    if not (tree_changed or pruned_any or const_improved):
        if ind.genome != old_genome:
            reMap_fn(ind, old_genome, grammar, max_tree_depth, codon_consumption)
            if old_fitness is not None:
                ind.fitness.values = (old_fitness,)
        return False

    final_genome = tree_to_genome(
        simplified_root,
        grammar,
        codon_size=codon_size,
        codon_consumption=codon_consumption,
        tail=clean_tail,
        rng=_rng,
    )
    reMap_fn(ind, final_genome, grammar, max_tree_depth, codon_consumption)
    if ind.invalid:
        reMap_fn(ind, old_genome, grammar, max_tree_depth, codon_consumption)
        if old_fitness is not None:
            ind.fitness.values = (old_fitness,)
        return False

    if evaluate_fn is not None and points_train is not None and old_fitness is not None:
        new_fit = evaluate_fn(ind, points_train)[0]
        minimize = ind.fitness.weights[0] < 0
        strictly_better = (not math.isnan(new_fit)) and (
            (new_fit < old_fitness - 1e-9) if minimize else (new_fit > old_fitness + 1e-9)
        )
        worse = math.isnan(new_fit) or ((new_fit > old_fitness + 1e-9) if minimize else (new_fit < old_fitness - 1e-9))
        # Anti-Clone Guard: If fitness only tied (did not strictly improve) and another
        # individual in the population already has this exact simplified phenotype,
        # keep the original stepping-stone structure to preserve population diversity!
        if worse or (not strictly_better and existing_phenotypes is not None and ind.phenotype in existing_phenotypes):
            reMap_fn(ind, old_genome, grammar, max_tree_depth, codon_consumption)
            ind.fitness.values = (old_fitness,)
            return False
        ind.fitness.values = (new_fit,)
        if existing_phenotypes is not None:
            existing_phenotypes.add(ind.phenotype)
    elif old_fitness is not None:
        ind.fitness.values = (old_fitness,)

    return True


def lamarckian_population_step(
    population: list,
    grammar,
    reMap_fn,
    evaluate_fn=None,
    points_train=None,
    max_tree_depth: int = 35,
    codon_size: int = 255,
    codon_consumption: str = "lazy",
    top_polish_fraction: float = 0.08,
    step_seed: int = 0,
) -> Tuple[int, int]:
    """
    Non-Destructive Lamarckian Elite Polishing & Bloat Compaction:
    Clones the top elites, applies algebraic/boolean simplification, causal knock-out
    pruning, and coordinate constant tuning on the clone, and injects improvements into
    the worst slots of the population so existing elite genomes/tails are never lost.
    """
    import copy

    valid = [
        ind for ind in population
        if not getattr(ind, "invalid", True)
        and ind.fitness.valid
        and not math.isnan(ind.fitness.values[0])
    ]
    if len(valid) < 6:
        return 0, 0

    local_rng = random.Random(9973 + step_seed)
    minimize = valid[0].fitness.weights[0] < 0
    sorted_valid = sorted(valid, key=lambda x: x.fitness.values[0], reverse=not minimize)
    n_polish = min(5, max(2, int(len(population) * top_polish_fraction)))
    elites_to_process = sorted_valid[:n_polish]
    worst_slots = list(reversed(sorted_valid[-n_polish:]))
    existing_phenotypes = {ind.phenotype for ind in valid}

    n_simplified = 0
    n_polished = 0
    worst_idx = 0

    for elite in elites_to_process:
        if worst_idx >= len(worst_slots):
            break
        probe = copy.deepcopy(elite)
        orig_fit = elite.fitness.values[0]
        orig_used = elite.used_codons

        changed = lamarckian_simplify_individual(
            probe,
            grammar,
            reMap_fn=reMap_fn,
            evaluate_fn=evaluate_fn,
            points_train=points_train,
            max_tree_depth=max_tree_depth,
            codon_size=codon_size,
            codon_consumption=codon_consumption,
            tune_constants=True,
            n_tune_steps=10,
            rng=local_rng,
            force_simplify=True,
            existing_phenotypes=existing_phenotypes,
        )
        if not changed or probe.invalid:
            continue

        new_fit = probe.fitness.values[0]
        strictly_better = (new_fit < orig_fit - 1e-9) if minimize else (new_fit > orig_fit + 1e-9)
        much_smaller_same_fit = (
            abs(new_fit - orig_fit) <= 1e-9 and probe.used_codons < orig_used * 0.75
        )

        if strictly_better or much_smaller_same_fit:
            target_worst = worst_slots[worst_idx]
            worst_idx += 1
            reMap_fn(target_worst, list(probe.genome), grammar, max_tree_depth, codon_consumption)
            if evaluate_fn is not None and points_train is not None:
                target_worst.fitness.values = evaluate_fn(target_worst, points_train)
            else:
                target_worst.fitness.values = (new_fit,)
            n_simplified += 1
            if strictly_better:
                n_polished += 1

    return n_simplified, n_polished
