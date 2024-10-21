"""Test the pyscal module to be used for testing.

These functions also need to be correct in order to trust tests."""

from contextlib import contextmanager

import pandas as pd
import pytest

from pyscal.utils.testing import float_df_checker, sat_table_str_ok


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "string, expectation",
    [
        pytest.param(None, pytest.raises(AssertionError)),
        pytest.param("", pytest.raises(AssertionError)),
        pytest.param(
            "SWOF\n 0 0 0\n/",
            pytest.raises(AssertionError),
            id="missing_pyscal_header",
        ),
        ("-- pyscal: \nSWOF\n0 0 0 0 \n/", does_not_raise()),
        pytest.param(
            "-- pyscal: \n\nSWOF\n0 0 0 0 \n/",
            does_not_raise(),
            id="whitespace_suffix_allowed",
        ),
        pytest.param(
            "-- pyscal: \n\nSWOF\n0 0 0\n/", does_not_raise(), id="three_numbers_ok"
        ),
        pytest.param(
            "-- pyscal: \n\nSWOF\n0 0\n/",
            pytest.raises(AssertionError),
            id="two_numbers_not_ok",
        ),
        ("-- pyscal: \nSGOF\n0 0 0 0 \n/", does_not_raise()),
        ("-- pyscal: \n-- foo\nSGOF\n0 0 0 0 \n/", does_not_raise()),
        pytest.param(
            "-- pyscal: \nRWOF\n0 0 0 0 \n/",
            pytest.raises(AssertionError),
            id="invalid_keyword",
        ),
        pytest.param(
            "-- pyscal: \nSROF\n0 0 0 0 \n/",
            does_not_raise(),
            id="invalid_keyword_that_passes",
        ),
        pytest.param(
            "-- pyscal: \nSWOF\n 0 0 0 0 \n/",
            pytest.raises(AssertionError),
            id="whitespace_prefix_disallowed",
        ),
        pytest.param(
            "-- pyscal: \nSWOF\na 0 0 0 \n/",
            pytest.raises(AssertionError),
            id="not_float",
        ),
        pytest.param(
            "-- pyscal: \nSWOF\n0 0 0 0 0\n/",
            pytest.raises(AssertionError),
            id="too_many_floats",
        ),
        pytest.param(
            "-- pyscal: \nSWOF\n-1 0 0 0\n/",
            pytest.raises(AssertionError),
            id="negative_number",
        ),
        pytest.param(
            "-- pyscal: \nSWOF\n0 0 -1 0\n/",
            pytest.raises(AssertionError),
            id="negative_number",
        ),
        pytest.param(
            "-- pyscal: \nSWOF\n0 0 -0 0\n/",
            pytest.raises(AssertionError),
            id="negative_zero",
        ),
        pytest.param(
            "-- pyscal: \nSWOF\n1.001 0 0 0\n/",
            pytest.raises(AssertionError),
            id="saturation_above_1",
        ),
        pytest.param(
            "-- pyscal: \nSWOF\n1 0 0 20.000000\n/",
            does_not_raise(),
            id="cap_pressure_can_be_larger",
        ),
        pytest.param(
            "-- pyscal: \nSWOF\n0 0 0 0\n1 1 1\n/",
            pytest.raises(AssertionError),
            id="non_rectangular_frame",
        ),
    ],
)
def test_sat_table_str_ok(string, expectation):
    """Test that we can determine whether a string is accepted by Eclipse

    (this is instead of actually testing in the simulator)
    """
    with expectation:
        sat_table_str_ok(string)


@pytest.mark.parametrize(
    "data, value, expected, expectation",
    [
        ([[0, 1], [1, 2]], 0, 1, does_not_raise()),
        pytest.param([[0, 1], [1, 2]], 0, 2, pytest.raises(AssertionError)),
        ([[0, 1], [1, 2]], 1, 2, does_not_raise()),
        ([[1e-07, 1], [1, 2]], 0, 1, does_not_raise()),
        ([[1e-06, 1], [1, 2]], 0, 1, does_not_raise()),
        ([[1e-05, 1], [1, 2]], 0, 1, does_not_raise()),
        ([[0.0001, 1], [1, 2]], 0, 1, does_not_raise()),
        ([[0.001, 1], [1, 2]], 0, 1, does_not_raise()),
        ([[0.01, 1], [1, 2]], 0, 1, does_not_raise()),
        ([[0.1, 1], [1, 2]], 0, 1, does_not_raise()),
        ([[-0.1, 1], [1, 2]], 0, 1, does_not_raise()),
        ([[-0.001, 1], [1, 2]], 0, 1, does_not_raise()),
        ([[0, 1], [0.1, 1.5], [1, 2]], 0.1, 1.5, does_not_raise()),
        ([[0.09, 1], [0.1, 1.5], [1, 2]], 0.1, 1.5, does_not_raise()),
        ([[0.099, 1], [0.1, 1.5], [1, 2]], 0.1, 1.5, does_not_raise()),
        ([[0.0999, 1], [0.1, 1.5], [1, 2]], 0.1, 1.5, does_not_raise()),
        ([[0.09999, 1], [0.1, 1.5], [1, 2]], 0.1, 1.5, does_not_raise()),
        ([[0.099999, 1], [0.1, 1.5], [1, 2]], 0.1, 1.5, does_not_raise()),
        ([[0.09999999, 1], [0.1, 1.5], [1, 2]], 0.1, 1.5, does_not_raise()),
        pytest.param([[0, 0], [1, 1]], 0, 1e-05, does_not_raise()),
        pytest.param(
            [[0, 0], [1, 1]],
            0,
            0.0001,
            pytest.raises(AssertionError),
            id="too_far_from_zero",
        ),
        pytest.param([[0, 0], [1, 1]], 1, 1.000001, does_not_raise()),
        pytest.param(
            [[0, 0], [1, 1]],
            1,
            1.0001,
            pytest.raises(AssertionError),
            id="too_far_from_one",
        ),
    ],
)
def test_float_df_checker(data, value, expected, expectation):
    """Test that we can lookup in a dataframe for values and test them"""
    with expectation:
        assert float_df_checker(
            pd.DataFrame(columns=["idx", "values"], data=data),
            "idx",
            value,
            "values",
            expected,
        )
