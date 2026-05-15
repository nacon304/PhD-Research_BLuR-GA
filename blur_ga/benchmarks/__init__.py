"""Synthetic benchmark generators for BLuR-GA evaluation."""

from .linkage_problems import (
    LinkageBenchmarkProblem,
    PairwiseQUBOProblem,
    IsingSpinGlassProblem,
    PlantedXORProblem,
    NKLandscapeProblem,
    MaxSATProblem,
    GametesStyleProblem,
    BenchmarkSpec,
    build_problem,
    write_problem,
    load_problem,
)

__all__ = [
    "LinkageBenchmarkProblem",
    "PairwiseQUBOProblem",
    "IsingSpinGlassProblem",
    "PlantedXORProblem",
    "NKLandscapeProblem",
    "MaxSATProblem",
    "GametesStyleProblem",
    "BenchmarkSpec",
    "build_problem",
    "write_problem",
    "load_problem",
]
