# -*- coding: utf-8 -*-
"""
Mechanic A: Derivation-Tree Tracing & Phenotype-Preserving Codon Remapping.

When the Epigenetic Agent updates the .bnf grammar (adding/reordering rules),
the number of choices k_A for non-terminal A changes. In standard GRAPE,
rule_idx = codon % k_A, so changing k_A scrambles existing genomes.

This module:
1. Parses any valid GRAPE Individual's genome into an exact BNF DerivationNode tree.
2. Remaps every consumed codon using:
       c'_i = k_new * ((c_i // k_old) % max_mult) + r_new
   so that c'_i % k_new == r_new and the individual's decoded phenotype is
   100% preserved under the upgraded grammar.
"""

import re
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

NT_PATTERN = re.compile(r"\<([\(\)\w,-.]+)\>")


@dataclass
class DerivationNode:
    """A single node in a BNF derivation tree."""
    nt_name: str
    nt_index: int
    rule_index: int
    rule_str: str
    codon_value: Optional[int]
    k_choices: int
    children: List["DerivationNode"] = field(default_factory=list)
    phenotype_cache: Optional[str] = None

    def render_phenotype(self) -> str:
        """Render the exact phenotype string produced by this subtree."""
        if not self.children:
            return self.rule_str
        out = self.rule_str
        for child in self.children:
            out = out.replace(child.nt_name, child.render_phenotype(), 1)
        return out

    def max_subtree_depth(self) -> int:
        """Return depth of this subtree (1 for terminal leaf expansion)."""
        if not self.children:
            return 1
        return 1 + max(c.max_subtree_depth() for c in self.children)

    def count_terminal_nodes(self) -> int:
        """Count arity-0 terminal expansions in this subtree (matches GRAPE nodes)."""
        if not self.children:
            return 1
        return sum(c.count_terminal_nodes() for c in self.children)

    def clone(self) -> "DerivationNode":
        return DerivationNode(
            nt_name=self.nt_name,
            nt_index=self.nt_index,
            rule_index=self.rule_index,
            rule_str=self.rule_str,
            codon_value=self.codon_value,
            k_choices=self.k_choices,
            children=[c.clone() for c in self.children],
            phenotype_cache=self.phenotype_cache,
        )


def build_derivation_tree(
    genome: List[int],
    grammar,
    max_depth: int = 35,
    codon_consumption: str = "lazy",
) -> Tuple[Optional[DerivationNode], int, List[int]]:
    """
    Build the exact left-most BNF derivation tree consumed by GRAPE's mapper.

    Returns:
        (root_node_or_None_if_invalid, used_codons, unconsumed_tail_codons)
    """
    idx_genome = 0
    start_nt = grammar.start_rule.strip()

    def _expand(nt_name: str, current_depth: int) -> Optional[DerivationNode]:
        nonlocal idx_genome
        if current_depth > max_depth:
            return None
        if nt_name not in grammar.non_terminals:
            return None
        nt_idx = grammar.non_terminals.index(nt_name)
        k_rules = grammar.n_rules[nt_idx]

        if codon_consumption == "lazy" and k_rules == 1:
            rule_idx = 0
            codon_val = None
        else:
            if idx_genome >= len(genome):
                return None
            codon_val = int(genome[idx_genome])
            rule_idx = codon_val % k_rules
            idx_genome += 1

        rule_entry = grammar.production_rules[nt_idx][rule_idx]
        rule_str = rule_entry[0]
        child_nts = ["<" + m + ">" for m in NT_PATTERN.findall(rule_str)]

        next_depth = current_depth + 1
        if next_depth > max_depth:
            return None

        children: List[DerivationNode] = []
        for child_nt in child_nts:
            child_node = _expand(child_nt, next_depth)
            if child_node is None:
                return None
            children.append(child_node)

        node = DerivationNode(
            nt_name=nt_name,
            nt_index=nt_idx,
            rule_index=rule_idx,
            rule_str=rule_str,
            codon_value=codon_val,
            k_choices=k_rules,
            children=children,
        )
        node.phenotype_cache = node.render_phenotype()
        return node

    root = _expand(start_nt, 1)
    if root is None:
        return None, 0, list(genome)

    tail = list(genome[idx_genome:])
    return root, idx_genome, tail


def tree_to_genome(
    root: DerivationNode,
    grammar,
    codon_size: int = 255,
    codon_consumption: str = "lazy",
    tail: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """
    Compile a DerivationNode tree back into a valid integer genome under `grammar`.
    Preserves high-order multiplier bits when a node already has `codon_value`.
    """
    _rng = rng if rng is not None else random
    codons: List[int] = []

    def _visit(node: DerivationNode) -> None:
        nt_idx = grammar.non_terminals.index(node.nt_name)
        k_new = grammar.n_rules[nt_idx]
        # Find rule index in target grammar by exact production string match
        prod_strings = [pr[0] for pr in grammar.production_rules[nt_idx]]
        if node.rule_str in prod_strings:
            r_new = prod_strings.index(node.rule_str)
        else:
            r_new = node.rule_index % k_new

        if codon_consumption == "lazy" and k_new == 1:
            pass
        else:
            max_mult = max(1, (codon_size + 1) // k_new)
            if node.codon_value is not None and node.k_choices > 0:
                mult = (node.codon_value // node.k_choices) % max_mult
            else:
                mult = _rng.randint(0, max_mult - 1)
            c_new = mult * k_new + r_new
            if c_new > codon_size:
                c_new = r_new
            codons.append(int(c_new))

        for child in node.children:
            _visit(child)

    _visit(root)

    if tail is not None and len(tail) > 0:
        codons.extend(tail)
    else:
        tail_len = max(1, len(codons) // 4)
        codons.extend(_rng.randint(0, codon_size) for _ in range(tail_len))

    return codons


def remap_individual_genome(
    ind,
    old_grammar,
    new_grammar,
    reMap_fn,
    max_tree_depth: int = 35,
    codon_size: int = 255,
    codon_consumption: str = "lazy",
) -> bool:
    """
    Adjust `ind.genome` in-place so that `ind` produces the exact same phenotype
    under `new_grammar` as it did under `old_grammar`.

    Returns True if phenotype was preserved identically, False if individual was invalid.
    """
    if getattr(ind, "invalid", True):
        reMap_fn(ind, ind.genome, new_grammar, max_tree_depth, codon_consumption)
        return False

    old_phenotype = ind.phenotype
    root, used, tail = build_derivation_tree(
        ind.genome, old_grammar, max_depth=max_tree_depth, codon_consumption=codon_consumption
    )
    if root is None:
        reMap_fn(ind, ind.genome, new_grammar, max_tree_depth, codon_consumption)
        return False

    new_genome = tree_to_genome(
        root,
        new_grammar,
        codon_size=codon_size,
        codon_consumption=codon_consumption,
        tail=tail,
    )
    saved_fitness = ind.fitness.values if ind.fitness.valid else None
    reMap_fn(ind, new_genome, new_grammar, max_tree_depth, codon_consumption)
    if saved_fitness is not None and ind.phenotype == old_phenotype and not ind.invalid:
        ind.fitness.values = saved_fitness
        return True
    return ind.phenotype == old_phenotype


def remap_population_genomes(
    population: list,
    old_grammar,
    new_grammar,
    reMap_fn,
    max_tree_depth: int = 35,
    codon_size: int = 255,
    codon_consumption: str = "lazy",
) -> Tuple[int, int]:
    """
    Remap all individuals in `population` from `old_grammar` to `new_grammar`.
    Returns (n_preserved, n_total_valid).
    """
    n_valid = 0
    n_preserved = 0
    for ind in population:
        if not getattr(ind, "invalid", True):
            n_valid += 1
            if remap_individual_genome(
                ind,
                old_grammar,
                new_grammar,
                reMap_fn=reMap_fn,
                max_tree_depth=max_tree_depth,
                codon_size=codon_size,
                codon_consumption=codon_consumption,
            ):
                n_preserved += 1
        else:
            reMap_fn(ind, ind.genome, new_grammar, max_tree_depth, codon_consumption)
    return n_preserved, n_valid
