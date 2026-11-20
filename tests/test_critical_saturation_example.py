"""Check the exported-table diagnostic used in the documentation."""

import json
import runpy
from pathlib import Path

import pytest


def test_critical_saturation_example():
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "examples"
        / "critical_saturation.py"
    )
    result = runpy.run_path(str(path))["critical_saturation_example"]()
    assert result["specified_swcr"] == pytest.approx(0.057)
    assert result["last_in_memory_zero_sw"] == pytest.approx(0.057)
    assert result["last_printed_zero_sw"] == pytest.approx(0.064)
    assert result["last_printed_sw_at_or_below_cutoff"] == pytest.approx(0.074)
    assert result["cutoff"] == 1e-6

    # Independently evaluate the Corey formula at the three diagnostic rows.
    def krw(sw):
        return ((sw - 0.057) / (1.0 - 0.057 - 0.239814)) ** 4

    assert 0.0 < krw(0.064) < 1e-7
    assert 1e-7 < krw(0.074) < 1e-6
    assert krw(0.084) > 1e-6
    assert json.loads(json.dumps(result, allow_nan=False)) == result
