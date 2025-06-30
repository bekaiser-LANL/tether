"""
Core submodule for the tether package.

This module provides the main benchmarking engine components:
- `Analyzer`: for evaluating benchmark results and applying LLM or human grading.
- `Grader`: for deterministic or LLM-based grading of responses.
- `Proctor`: for administering benchmarks to LLMs and collecting their outputs.
- Utility functions for working with benchmark files and metadata.

Intended for direct import via `from tether.core import Analyzer, Grader, Proctor`.
"""

from tether.core.analyzer import Analyzer
from tether.core.grader import Grader
from tether.core.proctor import Proctor
from tether.core.utils import QuestionBank

__all__ = [
    "Analyzer",
    "Grader",
    "Proctor",
    "QuestionBank",
]
