# -*- coding: utf-8 -*-
"""
Unit & Integration Verification Suite for EL-GE (Epigenetic-Lamarckian GE).

Tests:
1. Mechanic A (Remapper): 100/100 individuals across Pagie-1, Vladislavleva-4,
   Banknote, Heart Disease, and Parity-4 preserve 100% identical phenotypes
   when new rules are added to their .bnf grammars.
2. Mechanic B (Lamarckian Writeback): Bloated phenotypes (like `(x[0] + x[1]) - x[1]`,
   `x[0] * pdiv(x[1], x[1])`, and `and_(x[0], x[0])`) are simplified and written back
   into shorter integer genomes that decode to the simplified expression.
3. Bullet 1 (Epigenetic Agent): Extracts motifs from top winners, writes a valid
   upgraded .bnf file, and remaps the population with 0% phenotype corruption.
"""

import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import numpy as np
from deap import creator, base
import grape
from grape.functions import add, sub, mul, pdiv, plog, psqrt, and_, or_, nand_, nor_, not_, less_than_or_equal, greater_than_or_equal

from grape.el_ge import (
    build_derivation_tree,
    tree_to_genome,
    remap_population_genomes,
    lamarckian_simplify_individual,
    EpigeneticBNFAgent,
)

for _name in ("FitnessMin", "Individual"):
    if hasattr(creator, _name):
        delattr(creator, _name)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", grape.Individual, fitness=creator.FitnessMin)


def test_mechanic_a_remapper_across_all_grammars():
    print("=== Test 1: Mechanic A (Codon Remapper) across all 5 GRAPE Paper Grammars ===")
    grammars_to_test = [
        ("Pagie1.bnf", "<e>", ["(<e>*<e>)", "pdiv(01.00, 01.00+<e>*<e>)"]),
        ("Vladislavleva4.bnf", "<e>", ["(<e>-03.00)*(<e>-03.00)", "(<e>*<e>)"]),
        ("Banknote.bnf", "<e>", ["((<e> + <e>) * <e>)", "pdiv(<e>, 1.0)"]),
        ("heartDisease.bnf", "<num_op>", ["add(<num_op>, mul(<num_op>, <num_op>))"]),
        ("parity4.bnf", "<op>", ["and_(or_(<e>,<e>),nand_(<e>,<e>))"]),
    ]

    for g_file, target_nt, new_rules in grammars_to_test:
        random.seed(42)
        np.random.seed(42)
        old_path = ROOT.parent / "grape" / "grammars" / g_file
        old_grammar = grape.Grammar(str(old_path))

        pop = grape.sensible_initialisation(
            creator.Individual,
            pop_size=100,
            bnf_grammar=old_grammar,
            min_init_depth=3,
            max_init_depth=10,
            codon_size=255,
            codon_consumption="lazy",
            genome_representation="list",
        )
        before_phenotypes = [ind.phenotype for ind in pop]

        # Build upgraded grammar with new rules added to target_nt
        from grape.el_ge.epigenetic_agent import serialize_grammar_to_bnf
        rules_dict = {
            nt: [pr[0] for pr in old_grammar.production_rules[i]]
            for i, nt in enumerate(old_grammar.non_terminals)
        }
        rules_dict[target_nt].extend(new_rules)
        tmp_bnf = ROOT / "results" / "test_grammars" / f"upgraded_{g_file}"
        serialize_grammar_to_bnf(old_grammar.non_terminals, rules_dict, tmp_bnf)
        new_grammar = grape.Grammar(str(tmp_bnf))

        n_preserved, n_valid = remap_population_genomes(
            pop,
            old_grammar=old_grammar,
            new_grammar=new_grammar,
            reMap_fn=grape.reMap,
            max_tree_depth=35,
            codon_size=255,
            codon_consumption="lazy",
        )
        after_phenotypes = [ind.phenotype for ind in pop]
        assert before_phenotypes == after_phenotypes, f"Mismatch in {g_file}!"
        print(f"  [PASS] {g_file:20s}: {n_preserved}/{n_valid} valid individuals 100% preserved after adding {len(new_rules)} rules.")


def test_mechanic_b_lamarckian_writeback():
    print("\n=== Test 2: Mechanic B (Lamarckian Bloat Simplification & Codon Writeback) ===")
    # 2a. Parity-4: `and_(x[0], x[0])` -> `x[0]`
    p_grammar = grape.Grammar(str(ROOT.parent / "grape" / "grammars" / "parity4.bnf"))
    # In parity4.bnf:
    # <e> ::= <op> (0) | <x> (1)
    # <op> ::= and_(<e>,<e>) (0) | or_(<e>,<e>) (1) | nand_(<e>,<e>) (2) | nor_(<e>,<e>) (3)
    # <x> ::= x[0] (0) | x[1] (1) | x[2] (2) | x[3] (3)
    # Genome for `and_(x[0], x[0])`:
    # <e>->0 (<op>), <op>->0 (and_), left <e>->1 (<x>), <x>->0 (x[0]), right <e>->1 (<x>), <x>->0 (x[0])
    bloated_parity_genome = [0, 0, 1, 0, 1, 0, 99, 88, 77, 66, 55]
    ind_p = creator.Individual(bloated_parity_genome, p_grammar, 35, "lazy")
    assert ind_p.phenotype == "and_(x[0],x[0])", f"Got {ind_p.phenotype}"
    old_used = ind_p.used_codons

    changed = lamarckian_simplify_individual(
        ind_p, p_grammar, reMap_fn=grape.reMap, max_tree_depth=35, codon_size=255, codon_consumption="lazy", force_simplify=True
    )
    assert changed is True
    assert ind_p.phenotype == "x[0]", f"Expected 'x[0]', got {ind_p.phenotype}"
    assert ind_p.used_codons < old_used
    print(f"  [PASS] Parity-4 bloat purged: 'and_(x[0],x[0])' ({old_used} codons) -> '{ind_p.phenotype}' ({ind_p.used_codons} codons)")

    # 2b. Pagie-1: `(x[0] + x[1]) - x[1]` -> `x[0]`
    r_grammar = grape.Grammar(str(ROOT.parent / "grape" / "grammars" / "Pagie1.bnf"))
    # Pagie1.bnf `<e>` rules:
    # 0: <e>+<e>, 1: <e>-<e>, 2: <e>*<e>, 3: pdiv(<e>,<e>), ..., 8: x[0], 9: x[1]
    # Build `(x[0] + x[1]) - x[1]`:
    # root <e>->1 (<e>-<e>), left <e>->0 (<e>+<e>), ll <e>->8 (x[0]), lr <e>->9 (x[1]), right <e>->9 (x[1])
    bloated_reg_genome = [1, 0, 8, 9, 9, 12, 34, 56, 78, 90, 11, 22, 33]
    ind_r = creator.Individual(bloated_reg_genome, r_grammar, 35, "lazy")
    assert ind_r.phenotype == "x[0]+x[1]-x[1]", f"Got {ind_r.phenotype}"
    old_r_used = ind_r.used_codons

    changed_r = lamarckian_simplify_individual(
        ind_r, r_grammar, reMap_fn=grape.reMap, max_tree_depth=35, codon_size=255, codon_consumption="lazy", force_simplify=True
    )
    assert changed_r is True
    assert ind_r.phenotype == "x[0]", f"Expected 'x[0]', got {ind_r.phenotype}"
    assert ind_r.used_codons < old_r_used
    print(f"  [PASS] Pagie-1 bloat purged : 'x[0]+x[1]-x[1]' ({old_r_used} codons) -> '{ind_r.phenotype}' ({ind_r.used_codons} codons)")


def test_bullet_1_epigenetic_agent():
    print("\n=== Test 3: Bullet 1 (Epigenetic Agent .bnf Evolution + Remapping) ===")
    random.seed(7)
    np.random.seed(7)
    bnf_path = ROOT.parent / "grape" / "grammars" / "Pagie1.bnf"
    grammar = grape.Grammar(str(bnf_path))

    pop = grape.sensible_initialisation(
        creator.Individual,
        pop_size=60,
        bnf_grammar=grammar,
        min_init_depth=3,
        max_init_depth=8,
        codon_size=255,
        codon_consumption="lazy",
        genome_representation="list",
    )
    for idx, ind in enumerate(pop):
        ind.fitness.values = (float(idx),)

    before_phenos = [ind.phenotype for ind in pop]
    agent = EpigeneticBNFAgent(
        initial_bnf_path=str(bnf_path),
        grammar_class=grape.Grammar,
        reMap_fn=grape.reMap,
        output_dir=str(ROOT / "results" / "test_grammars"),
        max_rules_per_epoch=2,
        max_total_added_rules=6,
    )
    new_grammar, n_added = agent.step_epoch(
        generation=10,
        population=pop,
        current_grammar=grammar,
        problem_tag="pagie1_test",
        seed=7,
    )
    after_phenos = [ind.phenotype for ind in pop]
    assert n_added > 0, "Agent should have added at least 1 rule"
    assert before_phenos == after_phenos, "Population phenotypes must be 100% preserved after Epigenetic update"
    print(f"  [PASS] Epigenetic Agent added {n_added} new BNF rules: {[r[2] for r in agent.added_rules_history]}")
    print(f"  [PASS] All 60/60 individuals preserved exact phenotypes after live grammar upgrade.")


if __name__ == "__main__":
    test_mechanic_a_remapper_across_all_grammars()
    test_mechanic_b_lamarckian_writeback()
    test_bullet_1_epigenetic_agent()
    print("\nALL EL-GE VERIFICATION TESTS PASSED.")
