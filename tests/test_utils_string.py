"""Test module for pyscal.utils"""

import pandas as pd

from pyscal.utils.string import comment_formatter, df2str


def test_df2str():
    """Test handling of roundoff issues when printing dataframes

    See also test_gasoil.py::test_roundoff()
    """
    # Easy cases:
    assert df2str(pd.DataFrame(data=[0.1]), digits=1).strip() == "0.1"
    assert df2str(pd.DataFrame(data=[0.1]), digits=3).strip() == "0.100"
    assert df2str(pd.DataFrame(data=[0.01]), digits=3).strip() == "0.010"

    # Tricky ones:
    # This one should be rounded down:
    assert df2str(pd.DataFrame(data=[0.0034999]), digits=3).strip() == "0.003"

    # This number would be 0.0035 in IEE754 and would then be rounded up,
    # but the rounding and string production do not depend on IEE754:
    assert df2str(pd.DataFrame(data=[0.003499999999998]), digits=3).strip() == "0.003"


def test_comment_formatter():
    """Test the comment formatter

    This is also tested through hypothesis.text()
    in test_wateroil and test_gasoil, there is it tested
    through the use of tag formatting
    """
    assert comment_formatter(None) == "-- \n"
    assert comment_formatter("\n") == "-- \n"
    assert comment_formatter("\r\n") == "-- \n"
    assert comment_formatter("\r") == "-- \n"
    assert comment_formatter("foo") == "-- foo\n"
    assert comment_formatter("foo", prefix="gaa") == "gaafoo\n"
    assert comment_formatter("foo\nbar") == "-- foo\n-- bar\n"
