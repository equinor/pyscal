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
    assert df2str(pd.DataFrame(data=[0.1]), digits=3, roundlevel=3).strip() == "0.100"
    assert df2str(pd.DataFrame(data=[0.1]), digits=3, roundlevel=4).strip() == "0.100"
    assert df2str(pd.DataFrame(data=[0.1]), digits=3, roundlevel=5).strip() == "0.100"
    assert df2str(pd.DataFrame(data=[0.01]), digits=3, roundlevel=2).strip() == "0.010"

    # Here roundlevel will ruin the result:
    assert df2str(pd.DataFrame(data=[0.01]), digits=3, roundlevel=1).strip() == "0.000"

    # Tricky ones:
    # This one should be rounded down:
    assert df2str(pd.DataFrame(data=[0.0034999]), digits=3).strip() == "0.003"
    # But if we are on the 9999 side due to representation error, the
    # number probably represents 0.0035 so it should be rounded up
    assert (
        df2str(pd.DataFrame(data=[0.003499999999998]), digits=3, roundlevel=5).strip()
        == "0.004"
    )
    # If we round to more digits than we have in IEE754, we end up truncating:
    assert (
        df2str(pd.DataFrame(data=[0.003499999999998]), digits=3, roundlevel=20).strip()
        == "0.003"  # "Wrong" due to IEE754 truncation.
    )
    # If we round straight to out output, we are not getting the chance to correct for
    # the representation error:
    assert (
        df2str(pd.DataFrame(data=[0.003499999999998]), digits=3, roundlevel=3).strip()
        == "0.003"  # Wrong
    )
    # So roundlevel > digits
    assert (
        df2str(pd.DataFrame(data=[0.003499999999998]), digits=3, roundlevel=4).strip()
        == "0.004"
    )
    # But digits < roundlevel < 15 works:
    assert (
        df2str(pd.DataFrame(data=[0.003499999999998]), digits=3, roundlevel=14).strip()
        == "0.004"
    )
    assert (
        df2str(pd.DataFrame(data=[0.003499999999998]), digits=3, roundlevel=15).strip()
        == "0.003"  # Wrong
    )

    # Double rounding is a potential issue, as:
    assert round(0.0034445, 5) == 0.00344
    assert round(round(0.0034445, 6), 5) == 0.00345  # Wrong
    # So if pd.to_csv would later round instead of truncate, we could be victim
    # of this, having roundlevel >  digits + 1 would avoid that:
    assert round(round(0.0034445, 7), 5) == 0.00344
    # (this is the rationale for roundlevel > digits + 1)


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
