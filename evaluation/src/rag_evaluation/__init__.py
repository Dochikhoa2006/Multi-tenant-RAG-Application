"""Local-only evaluation types.

This package deliberately does not import Ragas at module import time. The
Ragas dependency is loaded only by the explicit local evaluator entry point.
"""

from .models import EvaluationRecord, EvaluationResult, MetricOutcome

__all__ = ["EvaluationRecord", "EvaluationResult", "MetricOutcome"]
__version__ = "0.1.0"

