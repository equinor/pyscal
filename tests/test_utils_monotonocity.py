"""Test module for monotonocity support functions in pyscal"""

import numpy as np
import pandas as pd

import pytest

from pyscal.utils.string import df2str
from pyscal.utils.monotonocity import (
    clip_accumulate,
    check_limits,
    rows_to_be_fixed,
    check_almost_monotone,
    validate_monotonocity_arg,
)


def test_df2str_monotone():
    """Test the monotonocity enforcement in df2str()

    This test function essentially tests the function
    utils/monotonicity.py::modify_dframe_monotonocity
    """

    # A constant nonzero column, makes no sense as capillary pressure
    # but still we ensure it runs in eclipse:
    assert (
        df2str(pd.DataFrame(data=[1, 1, 1]), digits=2, monotone_column=0)
        == "1.00\n0.99\n0.98\n"
    )
    assert (
        df2str(
            pd.DataFrame(data=[1, 1, 1]),
            digits=2,
            monotone_column=0,
            monotone_direction=-1,
        )
        == "1.00\n0.99\n0.98\n"
    )
    assert (
        df2str(
            pd.DataFrame(data=[1, 1, 1]),
            digits=2,
            monotone_column=0,
            monotone_direction="dec",
        )
        == "1.00\n0.99\n0.98\n"
    )
    assert (
        df2str(
            pd.DataFrame(data=[1, 1, 1]),
            digits=2,
            monotone_column=0,
            monotone_direction=1,
        )
        == "1.00\n1.01\n1.02\n"
    )
    assert (
        df2str(
            pd.DataFrame(data=[1, 1, 1]),
            digits=2,
            monotone_column=0,
            monotone_direction="inc",
        )
        == "1.00\n1.01\n1.02\n"
    )
    assert (
        df2str(pd.DataFrame(data=[1, 1, 1]), digits=7, monotone_column=0)
        == "1.0000000\n0.9999999\n0.9999998\n"
    )

    # For strict monotonicity we will introduce negativity:
    dframe = pd.DataFrame(data=[0.00001, 0.0, 0.0, 0.0], columns=["pc"])
    assert (
        df2str(dframe, monotone_column="pc")
        == "0.0000100\n0.0000000\n-0.0000001\n-0.0000002\n"
    )

    # Actual data that has occured:
    dframe = pd.DataFrame(
        data=[0.0000027, 0.0000026, 0.0000024, 0.0000024, 0.0000017], columns=["pc"]
    )
    assert (
        df2str(dframe, monotone_column="pc")
        == "0.0000027\n0.0000026\n0.0000024\n0.0000023\n0.0000017\n"
    )


@pytest.mark.parametrize(
    "series, monotonocity, expected",
    [
        (
            [0.00, 0.0002, 0.01, 0.010001, 0.0100001, 0.01, 0.99, 1.0001, 1.00],
            {0: {"sign": 1, "lower": 0, "upper": 1}},
            ["0.00", "0.00", "0.01", "0.02", "0.03", "0.04", "0.99", "1.00", "1.00"],
        ),
        (
            [0.02, 0.01, 0.01, 0.00, 0.00],
            {0: {"sign": -1, "lower": 0}},
            ["0.02", "0.01", "0.00", "0.00", "0.00"],
        ),
        (
            [0.02, 0.01, 0.01, 0.00, 0.00],
            {0: {"sign": -1}},
            ["0.02", "0.01", "-0.00", "-0.01", "-0.02"],
            #                 ^ this sign is optional, allowed when negative is allowed
        ),
        ([1.0, 1.0, 1.0], {0: {"sign": 1, "upper": 1}}, ["1.00", "1.00", "1.00"]),
        ([1, 1, 1], {0: {"sign": 1}}, ["1.00", "1.01", "1.02"]),
        ([1, 1, 1], {0: {"sign": -1, "upper": 1}}, ["1.00", "1.00", "1.00"]),
        ([1, 1, 1], {0: {"sign": -1}}, ["1.00", "0.99", "0.98"]),
        ([1, 1, 1], {0: {"sign": -1, "lower": 1}}, ["1.00", "1.00", "1.00"]),
        ([0, 0, 0], {0: {"sign": -1, "allowzero": True}}, ["0.00", "0.00", "0.00"]),
        ([0, 0, 0], {0: {"sign": 1, "allowzero": True}}, ["0.00", "0.00", "0.00"]),
        (
            [1, 1, 0.5, 0.01, 0.01, 0, 0],
            {0: {"sign": -1, "lower": 0, "upper": 1}},
            ["1.00", "1.00", "0.50", "0.01", "0.00", "0.00", "0.00"],
        ),
        (
            [0, 0, 0.01, 0.01, 0.5, 1, 1],
            {0: {"sign": 1, "lower": 0, "upper": 1}},
            ["0.00", "0.00", "0.01", "0.02", "0.50", "1.00", "1.00"],
        ),
        (
            [1, 0.01, 1e-3, 0],
            {0: {"sign": -1, "lower": 0, "upper": 1}},
            ["1.00", "0.01", "0.00", "0.00"],
        ),
        (
            [1, 0.01, 1e-9, 1e-10, 0],  # Tricky due to epsilon
            {0: {"sign": -1, "lower": 0, "upper": 1}},
            ["1.00", "0.01", "0.00", "0.00", "0.00"],
        ),
        (
            [1, 0.01, 4.66e-8, 1.56e-8, 4.09e-9, 1e-10, 0],  # Tricky due to epsilon
            {0: {"sign": -1, "lower": 0, "upper": 1}},
            ["1.00", "0.01", "0.00", "0.00", "0.00", "0.00", "0.00"],
        ),
        (
            [1, 0.01, 4.78e-8, 2.09e-8, 4.09e-9, 1e-10, 0],  # Tricky due to epsilon
            {0: {"sign": -1, "lower": 0, "upper": 1}},
            ["1.00", "0.01", "0.00", "0.00", "0.00", "0.00", "0.00"],
        ),
        # Test two columns at the same time:
        (
            [[0, 0], [0, 0], [0, 0]],
            {0: {"sign": -1, "allowzero": True}, 1: {"sign": -1, "allowzero": True}},
            ["0.00 0.00", "0.00 0.00", "0.00 0.00"],
        ),
        (
            [[1, 1], [1, 1], [1, 1]],
            {0: {"sign": -1}, 1: {"sign": 1}},
            ["1.00 1.00", "0.99 1.01", "0.98 1.02"],
        ),
        # Example in docstring for modify_dframe_monotonocity()
        (
            [0.00, 0.0002, 0.01, 0.010001, 0.0100001, 0.01, 0.99, 0.999, 1.0001, 1.00],
            {0: {"sign": 1, "lower": 0, "upper": 1}},
            [
                "0.00",
                "0.00",
                "0.01",
                "0.02",
                "0.03",
                "0.04",
                "0.99",
                "1.00",
                "1.00",
                "1.00",
            ],
        ),
    ],
)
def test_df2str_nonstrict_monotonocity(series, monotonocity, expected):
    """Test that we can have non-strict monotonocity at upper and/or lower limits"""
    assert (
        df2str(
            pd.DataFrame(data=series),
            digits=2,
            monotonocity=monotonocity,
        ).splitlines()
        == expected
    )


# Test similarly for digits=1:
@pytest.mark.parametrize(
    "series, monotonocity, expected",
    [
        (
            [0.00, 0.0002, 0.01, 0.010001, 0.0100001, 0.01, 0.99, 1.0001, 1.00],
            {0: {"sign": 1, "lower": 0, "upper": 1}},
            ["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "1.0", "1.0", "1.0"],
        ),
        (
            [0.2, 0.1, 0.1, 0.0, 0.0],
            {0: {"sign": -1, "lower": 0}},
            ["0.2", "0.1", "0.0", "0.0", "0.0"],
        ),
        (
            [0.2, 0.1, 0.1, 0.0, 0.0],
            {0: {"sign": -1}},
            ["0.2", "0.1", "-0.0", "-0.1", "-0.2"],
            #               ^ this sign is optional, allowed when negative is allowed
        ),
        ([1.0, 1.0, 1.0], {0: {"sign": 1, "upper": 1}}, ["1.0", "1.0", "1.0"]),
        ([1, 1, 1], {0: {"sign": 1}}, ["1.0", "1.1", "1.2"]),
        ([1, 1, 1], {0: {"sign": -1, "upper": 1}}, ["1.0", "1.0", "1.0"]),
        ([1, 1, 1], {0: {"sign": -1}}, ["1.0", "0.9", "0.8"]),
        ([1, 1, 1], {0: {"sign": -1, "lower": 1}}, ["1.0", "1.0", "1.0"]),
        ([0, 0, 0], {0: {"sign": -1, "allowzero": True}}, ["0.0", "0.0", "0.0"]),
        ([0, 0, 0], {0: {"sign": 1, "allowzero": True}}, ["0.0", "0.0", "0.0"]),
        (
            [1, 1, 0.5, 0.01, 0.01, 0, 0],
            {0: {"sign": -1, "lower": 0, "upper": 1}},
            ["1.0", "1.0", "0.5", "0.0", "0.0", "0.0", "0.0"],
        ),
        (
            [0, 0, 0.1, 0.1, 0.5, 1, 1],
            {0: {"sign": 1, "lower": 0, "upper": 1}},
            ["0.0", "0.0", "0.1", "0.2", "0.5", "1.0", "1.0"],
        ),
        (
            [1, 0.1, 1e-2, 0],
            {0: {"sign": -1, "lower": 0, "upper": 1}},
            ["1.0", "0.1", "0.0", "0.0"],
        ),
        # Example in docstring for modify_dframe_monotonocity()
        (
            [0.00, 0.0002, 0.01, 0.010001, 0.0100001, 0.01, 0.99, 0.999, 1.0001, 1.00],
            {0: {"sign": 1, "lower": 0, "upper": 1}},
            [
                "0.0",
                "0.0",
                "0.0",
                "0.0",
                "0.0",
                "0.0",
                "1.0",
                "1.0",
                "1.0",
                "1.0",
            ],
        ),
    ],
)
def test_df2str_nonstrict_monotonocity_digits1(series, monotonocity, expected):
    """Test that we can have non-strict monotonocity at upper and/or lower limits"""
    assert (
        df2str(
            pd.DataFrame(data=series),
            digits=1,
            monotonocity=monotonocity,
        ).splitlines()
        == expected
    )


@pytest.mark.parametrize(
    "series, monotonocity",
    [
        (
            [0, 1],
            {0: {"sign": -1}},
        ),
        ([0, 0.5, 1], {0: {"sign": -1}}),
        (
            [0, 1],
            {0: {"sign": -1, "lower": 0, "upper": 1}},
        ),
        ([0, 0.5, 1], {0: {"sign": -1, "lower": 0, "upper": 1}}),
        ([0, 1], {1: "foo"}),  # not a dict of dict
        ([0, 1], {0: {}}),  # sign is required
        ([0, 1], {0: {"sgn": 1}}),  # sign is required
        ([0, 1], {0: {"sign": 2}}),  # sign have abs = 1
        ([0, 1], {0: {"sign": 1, "lower": "a"}}),
        ([0, 1], {0: {"sign": 1, "lower": 2}}),
        ([0, 1], {0: {"sign": 1, "upper": -2}}),
        ([0], {0: {"sign": -1, "upper": -1}}),
        ([0], {0: {"sign": -1, "lower": 1}}),
        ([0], {0: {"sign": 1, "upper": -1}}),
        ([0], {0: {"sign": 1, "lower": 1}}),
    ],
)
def test_df2str_nonstrict_monotonocity_valueerror(series, monotonocity):
    """Check we get ValueError in the correct circumstances"""
    with pytest.raises(ValueError):
        df2str(
            pd.DataFrame(data=series),
            digits=2,
            monotonocity=monotonocity,
        )


@pytest.mark.parametrize(
    "series, monotonocity, expected_series",
    [
        ([], {"sign": 1}, []),
        ([0], {"sign": 1}, [0]),
        ([0, 0], {"sign": 1}, [0, 0]),
        ([0, -0.1], {"sign": 1}, [0, 0]),
        ([0, -0.1, -0.1], {"sign": 1}, [0, 0, 0]),
        ([0, -0.1], {"sign": -1}, [0, -0.1]),
        ([0, -0.1, -0.1], {"sign": -1}, [0, -0.1, -0.1]),
        (np.array([0, 2]), {"sign": 1, "upper": 1}, [0, 1]),
        ([0, 2], {"sign": 1, "upper": 1, "lower": 0.1}, [0.1, 1]),
        ([0, 2], {"sign": -1, "upper": 1, "lower": 0.1}, [0.1, 0.1]),
    ],
)
def test_clip_accumulate(series, monotonocity, expected_series):
    """Test that we are able to clip to upper and lower limits, and
    use numpy's accumulate to ensure non-strict monotonocity"""
    assert (clip_accumulate(series, monotonocity) == expected_series).all()


@pytest.mark.parametrize(
    "series, monotonocity, colname, error_str",
    [
        ([], {}, "", None),
        ([], {}, "foo", None),
        ([2], {"upper": 1}, "foobar", "larger than upper limit in column foobar"),
        ([2], {"upper": 1}, "", "larger than upper limit in column"),
        ([2], {"upper": 1}, None, "larger than upper limit in column None"),
        ([2], {"lower": 3}, "foobar", "smaller than lower limit in column foobar"),
    ],
)
def test_check_limits(series, monotonocity, colname, error_str):
    """Test that we can check upper and lower limits in series
    with proper error messages."""
    if error_str is not None:
        with pytest.raises(ValueError) as err:
            check_limits(series, monotonocity, colname)
        assert error_str in str(err)
    else:
        check_limits(series, monotonocity, colname)


@pytest.mark.parametrize(
    "series, monotonocity, digits, expected",
    [
        ([], {"sign": 1}, 0, []),
        ([], {"sign": 1}, 2, []),
        ([0], {"sign": 1}, 2, [False]),
        #
        ([0, 0.1], {"sign": 1}, 2, [False, False]),
        ([0, 0.1], {"sign": 1}, 1, [False, False]),
        ([0, 0.1], {"sign": 1}, 0, [False, True]),
        #
        ([0.1, 0], {"sign": -1}, 2, [False, False]),
        ([0.1, 0], {"sign": -2}, 1, [False, False]),
        ([0.1, 0], {"sign": -1}, 0, [False, True]),
        # But this is allowed if at limits:
        ([0, 0.1], {"sign": 2, "upper": 0.1}, 0, [False, False]),
        ([0.1, 0], {"sign": -1, "lower": 0}, 0, [False, False]),
    ],
)
def test_rows_to_be_fixed(series, monotonocity, digits, expected):
    """Check that we can make a boolean array of which elements must be fixed for
    monotonocity"""
    assert (rows_to_be_fixed(series, monotonocity, digits) == expected).all()


@pytest.mark.parametrize(
    "series, digits, sign, expected_error",
    [
        ([], 0, 1, None),
        ([0], 0, 1, None),
        ([0, -1], 0, 1, None),
        ([0, -1], 1, 1, None),  # Assumed constant with digits=1
        ([0, -1.001], 1, 1, ValueError),  # Just not constant enough
        ([0, -1], 2, 1, ValueError),
        ([0, 1], 2, -1, ValueError),
        ([0, 1], 1, -1, None),
    ],
)
def test_check_almost_monotone(series, digits, sign, expected_error):
    """Test that we have a way to determine if a series is almost monotone, that
    is with respect to how many digits is to be looked at."""
    if expected_error is not None:
        with pytest.raises(ValueError):
            check_almost_monotone(series, digits, sign)
    else:
        check_almost_monotone(series, digits, sign)


@pytest.mark.parametrize(
    "monotonocity, dframe_colnames, error_str",
    [
        ({}, [], None),
        ("sign", [], "monotonocity argument must be a dict"),
        ({"sign": 1}, [], "monotonocity argument must be a dict of dicts"),
        ({"foo": {"sign": 1}}, [], "Column foo does not exist in dataframe"),
        ({"foo": {"sign": 1}}, ["foo"], None),
        ({"foo": {"sgn": 1}}, ["foo"], "Unknown keys in monotonocity dict"),
        ({"foo": {"upper": 1}}, ["foo"], "Monotonocity sign not specified for foo"),
        ({"foo": {"sign": 1, "upper": 1}}, ["foo"], None),
        ({"foo": {"sign": 1, "lower": 1}}, ["foo"], None),
        ({"foo": {"sign": 1, "allowzero": True}}, ["foo"], None),
        (
            {"foo": {"sign": 1, "allowzero": "yes"}},
            ["foo"],
            "allowzero in monotonocity argument must be True/False",
        ),
        (
            {"foo": {"sign": "positive"}},
            ["foo"],
            "Monotonocity sign positive not valid",
        ),
        ({"foo": {"sign": 2}}, ["foo"], "Monotonocity sign must be -1 or +1"),
        ({"foo": {"sign": -2}}, ["foo"], "Monotonocity sign must be -1 or +1"),
    ],
)
def test_validate_monotonocity_arg(monotonocity, dframe_colnames, error_str):
    """Check error messages for monotonocity dictionaries"""
    if error_str is not None:
        with pytest.raises(ValueError) as err:
            validate_monotonocity_arg(monotonocity, dframe_colnames)
        assert error_str in str(err)
    else:
        validate_monotonocity_arg(monotonocity, dframe_colnames)
