# grape-bds (`v0.2.0`)

**GRAPE** (*Grammatical Algorithms in Python for Evolution*, BDS Research Group, University of Limerick) built on DEAP — now featuring **Epigenetic-Lamarckian Grammatical Evolution (EL-GE)** and full **NumPy 2.x** compatibility.

## Installation

```bash
pip install grape-bds
```

## What's New in `v0.2.0` — Epigenetic-Lamarckian Grammatical Evolution (`grape.el_ge`)

EL-GE extends classical Grammatical Evolution with three zero-disruption mechanisms that co-evolve the `.bnf` grammar and eliminate intron bloat during evolution:

1. **Mechanic A — Phenotype-Preserving Codon Remapping (`grape.el_ge.remapper`)**: Dynamically appends new production rules mid-run while remapping all population codons via the quotient-preserving modulo bijection $c'_i = k_{\text{new}} \cdot (\lfloor c_i / k_{\text{old}} \rfloor \bmod m_{\max}) + r_{\text{new}}$ (**100% phenotype preservation, zero mod-rule ripple**).
2. **Mechanic B — Lamarckian Derivation-Tree Writeback (`grape.el_ge.lamarckian`)**: Runs a 4-stage simplifier (algebraic/boolean identity reduction, causal subtree knock-out pruning, coordinate constant polishing, and epigenetic macro folding) and compiles compressed trees back into compact integer codons ($\Phi^{-1}$) with a $\ge 25$-codon reserve tail.
3. **Mechanic C — Dual-Oracle Epigenetic Rule Induction (`grape.el_ge.epigenetic_agent`)**:
   - `backend="ast"`: Zero-dependency deterministic AST subtree mining ($<5\text{ ms}$/epoch).
   - `backend="ollama"` / `"openai"` / `"gemini"`: Neuro-symbolic LLM hypothesis synthesis gated by a 5-stage syntactic and arity firewall.

## Quickstart

```python
import grape
from grape.el_ge import EpigeneticBNFAgent, ge_eaSimpleWithElitism_ELGE

# Drop-in replacement for grape.ge_eaSimpleWithElitism
agent = EpigeneticBNFAgent(
    initial_bnf_path="grammars/heartDisease.bnf",
    grammar_class=grape.Grammar,
    reMap_fn=grape.reMap,
    output_dir="evolved_grammars",
    backend="ast",  # or backend="ollama", model_name="gemma4:e4b"
)

population, logbook, final_grammar = ge_eaSimpleWithElitism_ELGE(
    population,
    toolbox,
    cxpb=0.8,
    mutpb=0.01,
    ngen=200,
    bnf_grammar=bnf_grammar,
    epigenetic_agent=agent,
    epigenetic_epoch=10,
    halloffame=hof,
    elite_size=1,
    max_tree_depth=35,
    max_init_tree_depth=13,
    codon_size=10000,
    codon_consumption="lazy",
    genome_representation="list",
)
```

## 120-Run Empirical Summary (10 Paired Seeds $\times$ 4 Benchmarks $\times$ 3 Arms)

| Benchmark | Arm | Test Mean Error | Paired Test W/T/L (Cliff's $\delta$) | Pop Mean Genome Len ($\Delta$ Bloat) | Mean Runtime / Seed |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`heartDisease`** | `GRAPE` (Baseline) | `0.2144` | — | `2,737.0` | `1,023.5s` |
| | `EL_GE_AST` | `0.2089` | `4W/1T/5L` (`+0.070`) | `1,356.5` (**-50.4%**) | **`469.1s` (2.18x)** |
| | `EL_GE_LLM` | **`0.1967`** | **`6W/0T/4L` (`+0.240`)** | **`1,246.1` (-54.5%)** | `897.8s` (**1.14x**) |
| **`pagie1`** | `GRAPE` (Baseline) | `0.0631` | — | `2,198.8` | `27.6s` |
| | `EL_GE_AST` | `0.0622` | `4W/3T/3L` (`+0.090`) | **`257.1` (-88.3%)** | **`18.1s` (1.52x)** |
| | `EL_GE_LLM` | **`0.0456`** | **`5W/1T/4L` (`+0.240`)** | `320.2` (**-85.4%**) | `91.2s` |
| **`vladislavleva4`** | `GRAPE` (Baseline) | **`0.0284`** | — | `1,282.3` | **`33.7s`** |
| | `EL_GE_AST` | `0.0329` | `3W/2T/5L` (`-0.020`) | `267.4` (**-79.1%**) | `33.8s` |
| | `EL_GE_LLM` | `0.0298` | **`5W/2T/3L`** (`+0.000`) | **`262.5` (-79.5%)** | `82.7s` |
| **`parity4`** | `GRAPE` (Baseline) | **`0.2438`** | — | `493.1` | `11.6s` |
| | `EL_GE_AST` | `0.2500` | `4W/1T/5L` (`-0.080`) | **`390.6` (-20.8%)** | **`8.9s` (1.30x)** |

## Credits & License
Built on **GRAPE** by Allan de Lima, Samuel Carvalho, Douglas Mota Dias, Enrique Naredo, Joseph P. Sullivan, and Conor Ryan (BDS Research Group, University of Limerick; *Signals* 2022). Distributed under the BSD 3-Clause License.
