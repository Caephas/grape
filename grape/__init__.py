"""GRAPE (Grammatical Algorithms in Python for Evolution) + EL-GE (v0.2.2)."""

from .grape import *  # noqa: F401, F403
from .functions import *  # noqa: F401, F403
from .algorithms import *  # noqa: F401, F403
from . import el_ge  # noqa: F401
from .el_ge import (  # noqa: F401
    EpigeneticBNFAgent,
    ge_eaSimpleWithElitism_ELGE,
    lamarckian_population_step,
    lamarckian_simplify_individual,
    remap_individual_genome,
    remap_population_genomes,
    selEpsilonLexicase,
)

__version__ = "0.2.2"
