"""
Feature engineering utilities for Soccer TIPS.

Current implementation scope: Layer 2 v0.1 feature-engineering foundation.

This package consumes Layer 1 canonical outputs and builds reusable frame-level,
phase-level, and match-level feature tables. It does not perform Layer 3 tactical
modeling, AI interpretation, dashboards, recommendations, or reporting.
"""

__all__ = [
    "layer2_schemas",
    "phase_context",
    "orientation",
    "team_shape",
    "aggregation",
    "build_layer2",
    "qa_layer2_outputs",
]
