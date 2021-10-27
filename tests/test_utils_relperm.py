"""Test module for relperm processing support code in pyscal"""

import numpy as np
import pandas as pd
import pytest

from pyscal.utils.relperm import crosspoint, estimate_diffjumppoint, truncate_zeroness

# pyscal.utils.relperm.crosspoint() is also tested in test_wateroil and test_gasoil.


@pytest.mark.parametrize(
    "value, zeronesslimit, expected",
    [(0, 0.1, 0), (0.01, 0.1, 0), (0.1, 0.1, 0.1), (1, 0.1, 1)],
)
def test_truncate_zeroness(value, zeronesslimit, expected):
    """Test truncation of zeroness, that is that numbers close to zero
    end up as zero, but others don't"""
    assert truncate_zeroness(value, zeronesslimit=zeronesslimit) == expected


@pytest.mark.parametrize(
    "data, expected",
    [
        pytest.param([[]], -1, id="empty_data"),
        pytest.param(
            [[0, 0, 0]],
            -1,
            id="too_few_rows",
        ),
        pytest.param(
            [[0, 0, 0], [0, 0, 0]],
            0.0,
            id="undefined_crosspoint",
            # Solution is not well defined, but zero is an ok answer.
        ),
        pytest.param(
            [[0, 0, 0], [1, 1, 1]],
            0,  # Should this be detected and -1 returned?
            id="identical_lines",
        ),
        pytest.param(
            [[1, 1, 1], [0, 0, 0]],
            1,  # Should this be detected and -1 returned?
            id="identical_lines",
        ),
        pytest.param(
            [[0, 0, 1], [1, 1, 2]],
            -1,
            id="parallel_lines",
        ),
        pytest.param(
            [[0, 0, 3], [1, 1, 2]],
            -1,
            id="not_crossing",
        ),
        pytest.param(
            [[0, np.nan, 1], [1, 1, 0]],
            -1,
            id="nan_input",
        ),
        pytest.param([[np.nan, 0, 1], [1, 1, 0]], -1, id="nan_input"),
        ([[0, 0, 1], [1, 1, 0]], 0.5),
    ],
)
def test_crosspoint(data, expected):
    """Test the crosspoint computation code.

    This is also tested for relevant data in test_wateroil.py and test_gasoil.py,
    here the possible error scenarios are more explored."""
    if data and data[0]:
        dframe = pd.DataFrame(columns=["A", "B", "C"], data=data)
    else:
        dframe = pd.DataFrame()
    assert np.isclose(
        crosspoint(dframe, "A", "B", "C"),
        expected,
    )


def test_diffjumppoint():
    """Test estimator for the jump in first derivative for some manually set up cases.

    This code is also extensively tested through test_fromtable"""

    dframe = pd.DataFrame(columns=["x", "y"], data=[[0, 0], [0.3, 0.2], [1, 1]])

    assert estimate_diffjumppoint(dframe, side="right") == 0.3
    assert estimate_diffjumppoint(dframe, side="left") == 0.3

    with pytest.raises(AssertionError):
        estimate_diffjumppoint(dframe, side="front")
    with pytest.raises(ValueError):
        estimate_diffjumppoint(dframe, side=None)

    dframe = pd.DataFrame(columns=["x", "y"], data=[[0, 0], [1, 1]])
    # We don't really care what gets printed from this, just don't crash..
    assert 0 <= estimate_diffjumppoint(dframe, side="right") <= 1
    assert 0 <= estimate_diffjumppoint(dframe, side="left") <= 1

    dframe = pd.DataFrame(
        columns=["x", "y"],
        data=[
            [0, 0],
            [0.1, 0.1],
            [0.2, 0.2],  # Linear until here
            [0.3, 0.4],  # Nonlinear region
            [0.4, 0.45],  # Nonlinear region
            [0.7, 0.7],  # Linear from here
            [0.8, 0.8],
            [1, 1],
        ],
    )
    assert estimate_diffjumppoint(dframe, side="left") == 0.2
    assert estimate_diffjumppoint(dframe, side="right") == 0.7

    dframe = pd.DataFrame(
        columns=["x", "y"],
        data=[
            [0, 0],
            [0.1, 0.0],
            [0.2, 0.0],  # Linear until here
            [0.3, 0.4],  # Nonlinear region
            [0.9, 1],  # Start linear region again
            [1, 1],
        ],
    )
    assert estimate_diffjumppoint(dframe, side="left") == 0.2
    assert estimate_diffjumppoint(dframe, side="right") == 0.9
