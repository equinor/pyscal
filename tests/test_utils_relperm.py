"""Test module for relperm processing support code in pyscal"""

import pandas as pd


from pyscal.utils.relperm import estimate_diffjumppoint

# pyscal.utils.relperm.crosspoint() is tested by test_wateroil and test_gasoil.


def test_diffjumppoint():
    """Test estimator for the jump in first derivative for some manually set up cases.

    This code is also extensively tested throuth test_addfromtable"""

    dframe = pd.DataFrame(columns=["x", "y"], data=[[0, 0], [0.3, 0.2], [1, 1]])

    assert estimate_diffjumppoint(dframe, side="right") == 0.3
    assert estimate_diffjumppoint(dframe, side="left") == 0.3

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
