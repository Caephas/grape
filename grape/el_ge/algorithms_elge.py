# -*- coding: utf-8 -*-
"""
Main Evolution Loop for Epigenetic-Lamarckian Grammatical Evolution (EL-GE).

Drop-in replacement for `grape_upstream/algorithms.py:ge_eaSimpleWithElitism`
that preserves 100% of the original GRAPE statistics and `REPORT_ITEMS` logging
while adding:
  - `use_lamarckian` (Mechanic B: Derivation-tree bloat simplification & writeback)
  - `use_epigenetic` (Bullet 1 + Mechanic A: Live .bnf grammar evolution + codon remapping)
  - `selEpsilonLexicase` (MAD epsilon-lexicase selection alongside tournament selection)
"""

import math
import random
import time
import warnings
from typing import Optional

import numpy as np
from deap import tools

from .epigenetic_agent import EpigeneticBNFAgent
from .lamarckian import lamarckian_population_step
from .remapper import remap_individual_genome


def selEpsilonLexicase(individuals: list, k: int) -> list:
    """
    MAD epsilon-lexicase selection over `ind.fitness_each_sample` (lower error is better).
    Compatible with GRAPE regression, classification, and boolean benchmarks.
    """
    pool = [
        ind for ind in individuals
        if not getattr(ind, "invalid", True)
        and getattr(ind, "fitness_each_sample", None) is not None
        and not math.isnan(ind.fitness.values[0])
    ]
    if not pool:
        valid_fallback = [ind for ind in individuals if not getattr(ind, "invalid", True)]
        src = valid_fallback if valid_fallback else individuals
        return [random.choice(src) for _ in range(k)]

    matrix = np.asarray([ind.fitness_each_sample for ind in pool], dtype=float)
    # Replace NaNs/Infs with large error
    matrix = np.nan_to_num(matrix, nan=1e9, posinf=1e9, neginf=1e9)
    medians = np.median(matrix, axis=0)
    epsilons = np.median(np.abs(matrix - medians), axis=0)

    minimize = pool[0].fitness.weights[0] < 0
    selected = []
    n_cases = matrix.shape[1]

    for _ in range(k):
        candidates = np.arange(len(pool))
        cases = list(range(n_cases))
        random.shuffle(cases)
        for case in cases:
            if len(candidates) <= 1:
                break
            if minimize:
                best = matrix[candidates, case].min()
                candidates = candidates[matrix[candidates, case] <= best + epsilons[case] + 1e-12]
            else:
                best = matrix[candidates, case].max()
                candidates = candidates[matrix[candidates, case] >= best - epsilons[case] - 1e-12]
        chosen_idx = int(random.choice(list(candidates)))
        selected.append(pool[chosen_idx])
    return selected


def varAnd_ELGE(
    population,
    toolbox,
    cxpb,
    mutpb,
    bnf_grammar,
    codon_size,
    max_tree_depth,
    codon_consumption,
    genome_representation,
    max_genome_length,
):
    """Standard GRAPE varAnd using the current (potentially evolved) `bnf_grammar`."""
    offspring = [toolbox.clone(ind) for ind in population]

    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(
                offspring[i - 1],
                offspring[i],
                bnf_grammar,
                max_tree_depth,
                codon_consumption,
                genome_representation,
                max_genome_length,
            )

    for i in range(len(offspring)):
        (offspring[i],) = toolbox.mutate(
            offspring[i],
            mutpb,
            codon_size,
            bnf_grammar,
            max_tree_depth,
            codon_consumption,
            max_genome_length,
        )

    return offspring


def ge_eaSimpleWithElitism_ELGE(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    elite_size,
    bnf_grammar,
    codon_size,
    max_tree_depth,
    max_genome_length=None,
    points_train=None,
    points_test=None,
    codon_consumption="lazy",
    report_items=None,
    genome_representation="list",
    stats=None,
    halloffame=None,
    verbose=False,
    # --- EL-GE extensions ---
    use_lamarckian: bool = False,
    use_epigenetic: bool = False,
    epigenetic_freq: int = 10,
    epigenetic_agent: Optional[EpigeneticBNFAgent] = None,
    reMap_fn=None,
    problem_tag: str = "benchmark",
    seed: int = 0,
    test_eval_every: int = 1,
):
    """
    Drop-in replacement for GRAPE's `ge_eaSimpleWithElitism` supporting:
      - Standard GRAPE (`use_lamarckian=False, use_epigenetic=False`)
      - Lamarckian GE (`use_lamarckian=True, use_epigenetic=False`)
      - Epigenetic GE (`use_lamarckian=False, use_epigenetic=True`)
      - Full EL-GE (`use_lamarckian=True, use_epigenetic=True`)
    """
    if report_items is None:
        report_items = []
    logbook = tools.Logbook()

    base_fields = (
        ["gen", "invalid"]
        + (stats.fields if stats else [])
        + (["fitness_test"] if points_test else [])
        + [
            "best_ind_length",
            "avg_length",
            "best_ind_nodes",
            "avg_nodes",
            "best_ind_depth",
            "avg_depth",
            "avg_used_codons",
            "best_ind_used_codons",
            "behavioural_diversity",
            "structural_diversity",
            "fitness_diversity",
            "selection_time",
            "generation_time",
            "lamarckian_simplified",
            "epigenetic_rules_added",
        ]
    )
    logbook.header = base_fields

    start_gen = time.time()
    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind, points_train)

    lam_count = 0

    def _compile_gen_metrics(gen_idx: int, sel_time: float, gen_start: float, lam_c: int, epi_c: int):
        valid0 = [ind for ind in population if not ind.invalid]
        valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]
        invalid = len(population) - len(valid0)

        list_structures = [str(ind.structure) for ind in valid]
        unique_structures = np.unique(list_structures) if list_structures else []
        structural_diversity = len(unique_structures) / len(population) if population else 0.0

        fitness_diversity = 0.0
        if "fitness_diversity" in report_items and valid and points_train is not None:
            unique_fit = np.unique([str(ind.fitness.values[0]) for ind in valid])
            fitness_diversity = len(unique_fit) / (len(points_train[1]) + 1)

        behavioural_diversity = 0.0
        if "behavioural_diversity" in report_items and valid and getattr(valid[0], "fitness_each_sample", None) is not None:
            behaviours = np.asarray([ind.fitness_each_sample for ind in valid], dtype=float)
            unique_behav = np.unique(behaviours, axis=0)
            behavioural_diversity = len(unique_behav) / len(population)

        best_ind_length = best_ind_nodes = best_ind_depth = best_ind_used_codons = 0
        fitness_test = np.nan
        if halloffame is not None and valid:
            halloffame.update(valid)
            best_ind = halloffame.items[0]
            best_ind_length = len(best_ind.genome)
            best_ind_nodes = best_ind.nodes
            best_ind_depth = best_ind.depth
            best_ind_used_codons = best_ind.used_codons
            if points_test is not None and (gen_idx == ngen or gen_idx % test_eval_every == 0):
                fitness_test = toolbox.evaluate(best_ind, points_test)[0]

        if valid:
            avg_length = sum(len(ind.genome) for ind in valid) / len(valid)
            avg_nodes = sum(ind.nodes for ind in valid) / len(valid)
            avg_depth = sum(ind.depth for ind in valid) / len(valid)
            avg_used_codons = sum(ind.used_codons for ind in valid) / len(valid)
        else:
            avg_length = avg_nodes = avg_depth = avg_used_codons = 0.0

        gen_time = time.time() - gen_start
        record = stats.compile(valid) if (stats and valid) else {}
        row = dict(
            gen=gen_idx,
            invalid=invalid,
            **record,
            best_ind_length=best_ind_length,
            avg_length=avg_length,
            best_ind_nodes=best_ind_nodes,
            avg_nodes=avg_nodes,
            best_ind_depth=best_ind_depth,
            avg_depth=avg_depth,
            avg_used_codons=avg_used_codons,
            best_ind_used_codons=best_ind_used_codons,
            behavioural_diversity=behavioural_diversity,
            structural_diversity=structural_diversity,
            fitness_diversity=fitness_diversity,
            selection_time=sel_time,
            generation_time=gen_time,
            lamarckian_simplified=lam_c,
            epigenetic_rules_added=epi_c,
        )
        if points_test is not None:
            row["fitness_test"] = fitness_test
        logbook.record(**row)
        if verbose:
            print(
                f"gen={gen_idx:3d} | min={record.get('min', float('nan')):.5f} "
                f"| test={fitness_test:.5f} | nodes={avg_nodes:.1f} | len={avg_length:.1f} "
                f"| lam={lam_c} | epi={epi_c}"
            )
        return valid

    valid = _compile_gen_metrics(0, 0.0, start_gen, lam_count, 0)

    for gen in range(1, ngen + 1):
        start_gen = time.time()
        epi_added = 0

        # Epigenetic BNF Grammar Update every `epigenetic_freq` generations
        if (
            use_epigenetic
            and epigenetic_agent is not None
            and reMap_fn is not None
            and gen % epigenetic_freq == 0
            and gen < ngen
        ):
            old_grammar = bnf_grammar
            bnf_grammar, epi_added = epigenetic_agent.step_epoch(
                generation=gen,
                population=population,
                current_grammar=bnf_grammar,
                problem_tag=problem_tag,
                seed=seed,
                max_tree_depth=max_tree_depth,
                codon_size=codon_size,
                codon_consumption=codon_consumption,
                evaluate_fn=toolbox.evaluate,
                points_train=points_train,
            )
            if epi_added > 0 and halloffame is not None:
                _rng_state = random.getstate()
                for hof_ind in halloffame.items:
                    remap_individual_genome(
                        hof_ind,
                        old_grammar=old_grammar,
                        new_grammar=bnf_grammar,
                        reMap_fn=reMap_fn,
                        max_tree_depth=max_tree_depth,
                        codon_size=codon_size,
                        codon_consumption=codon_consumption,
                    )
                random.setstate(_rng_state)
            valid0 = [ind for ind in population if not ind.invalid]
            valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]

        # Selection
        sel_start = time.time()
        offspring = toolbox.select(valid, len(population) - elite_size)
        selection_time = time.time() - sel_start

        # Crossover & Mutation
        offspring = varAnd_ELGE(
            offspring,
            toolbox,
            cxpb,
            mutpb,
            bnf_grammar,
            codon_size,
            max_tree_depth,
            codon_consumption,
            genome_representation,
            max_genome_length,
        )

        # Evaluate new offspring
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind, points_train)

        # Lamarckian step (Mechanic B) after initial 5-gen warmup so constant-only
        # trees at Gen 0 do not prematurely dominate tournament selection
        lam_count = 0
        if use_lamarckian and reMap_fn is not None and gen >= 5:
            _rng_state = random.getstate()
            lam_count, _ = lamarckian_population_step(
                offspring,
                bnf_grammar,
                reMap_fn=reMap_fn,
                evaluate_fn=toolbox.evaluate,
                points_train=points_train,
                max_tree_depth=max_tree_depth,
                codon_size=codon_size,
                codon_consumption=codon_consumption,
                step_seed=seed * 1000 + gen,
            )
            random.setstate(_rng_state)

        population[:] = offspring
        if halloffame is not None:
            for i in range(min(elite_size, len(halloffame.items))):
                population.append(toolbox.clone(halloffame.items[i]))

        valid = _compile_gen_metrics(gen, selection_time, start_gen, lam_count, epi_added)

    return population, logbook, bnf_grammar
