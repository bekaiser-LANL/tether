from . import analyzer
from . import grader
from . import generator
from . import utils
from . import proctor
from .benchmarks import standard_deviation
from .benchmarks import mediated_causality
from .benchmarks import simple_inequality
from setuptools_scm import get_version

__version__ = get_version(root="..", relative_to=__file__)

__all__ = [
    "analyzer",
    "grader",
    "proctor",
    "generator",    
    "standard_deviation",
    "mediated_causality",
    "simple_inequality",
    "utils"
]