"""Unit tests for the bench harness helpers. Skips anything that needs GPU."""

from __future__ import annotations

import numpy as np

from bench import parity_nmi  # noqa: E402


def test_parity_nmi_perfect_agreement():
    cpu = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    gpu = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    assert parity_nmi(cpu, gpu) == 1.0


def test_parity_nmi_relabeling_is_invariant():
    """NMI is label-permutation invariant — 0/1/2 vs 5/6/7 must score 1.0."""
    cpu = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    gpu = np.array([5, 5, 6, 6, 7, 7], dtype=np.int32)
    assert parity_nmi(cpu, gpu) == 1.0


def test_parity_nmi_excludes_outliers_from_both_sides():
    """Mismatched outliers should drop, not penalise."""
    cpu = np.array([0, 0, 1, 1, -1, -1], dtype=np.int32)
    gpu = np.array([0, 0, 1, 1, 0, 1], dtype=np.int32)
    # After masking the -1s on the CPU side, the remaining 4 pairs agree
    # perfectly, so NMI should be 1.0.
    assert parity_nmi(cpu, gpu) == 1.0


def test_parity_nmi_returns_nan_when_no_overlap():
    cpu = np.array([-1, -1, -1], dtype=np.int32)
    gpu = np.array([0, 1, 2], dtype=np.int32)
    result = parity_nmi(cpu, gpu)
    assert result != result  # NaN
