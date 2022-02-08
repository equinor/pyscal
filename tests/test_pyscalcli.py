"""Test the pyscal client"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from pyscal import pyscalcli
from pyscal.utils.testing import sat_table_str_ok


@pytest.mark.integration
def test_installed():
    """Test that the command line client is installed in PATH and
    starts up nicely"""
    assert subprocess.check_output(["pyscal", "-h"])


@pytest.mark.skipif(sys.platform == "win32", reason="UTF-8 problems on Windows")
@pytest.mark.skipif(sys.version_info < (3, 7), reason="Requires Python 3.7 or higher")
@pytest.mark.parametrize("verbosity_flag", [None, "--verbose", "--debug"])
def test_log_levels(tmp_path, verbosity_flag):
    """Test that we can control the log level from the command line
    client, and get log output from modules deep down"""

    relperm_file = str(
        # A cell in this xlsx contains "Åre 1.5", does not work on Windows
        Path(__file__).absolute().parent
        / "data"
        / "relperm-input-example.xlsx"
    )

    commands = ["pyscal", relperm_file]
    if verbosity_flag is not None:
        commands.append(verbosity_flag)

    result = subprocess.run(commands, cwd=tmp_path, capture_output=True, check=True)
    output = result.stdout.decode() + result.stderr.decode()

    if verbosity_flag is None:
        assert "INFO:" not in output
        assert "DEBUG:" not in output
    elif verbosity_flag == "--verbose":
        assert "INFO:" in output
        assert "DEBUG:" not in output
        assert "Loaded input data" in output
        assert "Keywords SWOF, SGOF (family 1) for 3 SATNUMs generated" in output
    elif verbosity_flag == "--debug":
        assert "INFO:" in output
        assert "DEBUG:" in output
        assert "Initialized GasOil with" in output
        assert "Initialized WaterOil with" in output
    else:
        raise ValueError("Unknown value for 'verbosity_flag'")


def test_pyscal_client_static(tmp_path, caplog, default_loglevel, mocker):
    # pylint: disable=unused-argument
    # default_loglevel fixture is in conftest.py
    """Test pyscal client for static relperm input"""
    testdir = Path(__file__).absolute().parent
    relperm_file = testdir / "data/relperm-input-example.xlsx"

    os.chdir(tmp_path)

    caplog.clear()
    mocker.patch("sys.argv", ["pyscal", str(relperm_file)])
    pyscalcli.main()
    assert Path("relperm.inc").is_file()
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    assert not any(record.levelno == logging.INFO for record in caplog.records)

    # We get one warning due to empty cells in xlsx:
    assert sum(record.levelno == logging.WARNING for record in caplog.records) == 1

    relpermlines = os.linesep.join(
        Path("relperm.inc").read_text(encoding="utf8").splitlines()
    )
    assert "SWOF" in relpermlines
    assert "SGOF" in relpermlines
    assert "SLGOF" not in relpermlines
    assert "SOF3" not in relpermlines
    sat_table_str_ok(relpermlines)

    caplog.clear()
    mocker.patch(
        "sys.argv", ["pyscal", str(relperm_file), "--output", "alt2relperm.inc"]
    )
    pyscalcli.main()
    assert Path("alt2relperm.inc").is_file()
    assert not any(record.levelno == logging.ERROR for record in caplog.records)

    caplog.clear()
    mocker.patch("sys.argv", ["pyscal", str(relperm_file), "-o", "altrelperm.inc"])
    pyscalcli.main()
    assert Path("altrelperm.inc").is_file()
    assert not any(record.levelno == logging.ERROR for record in caplog.records)

    caplog.clear()
    mocker.patch(
        "sys.argv", ["pyscal", str(relperm_file), "--family2", "-o", "relperm-fam2.inc"]
    )
    pyscalcli.main()
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    assert Path("relperm-fam2.inc").is_file()
    relpermlines = os.linesep.join(
        Path("relperm-fam2.inc").read_text(encoding="utf8").splitlines()
    )
    assert "SWFN" in relpermlines
    assert "SGFN" in relpermlines
    assert "SOF3" in relpermlines
    assert "SWOF" not in relpermlines
    assert "SGOF" not in relpermlines
    sat_table_str_ok(relpermlines)

    caplog.clear()
    mocker.patch(
        "sys.argv",
        ["pyscal", str(relperm_file), "--slgof", "--output", "relperm-slgof.inc"],
    )
    pyscalcli.main()
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    assert Path("relperm-slgof.inc").is_file()
    relpermlines = os.linesep.join(
        Path("relperm-slgof.inc").read_text(encoding="utf8").splitlines()
    )
    assert "SWOF" in relpermlines
    assert "SGOF" not in relpermlines
    assert "SLGOF" in relpermlines
    assert "SOF3" not in relpermlines
    sat_table_str_ok(relpermlines)

    caplog.clear()
    # Dump to deep directory structure that does not exists
    mocker.patch(
        "sys.argv",
        [
            "pyscal",
            str(relperm_file),
            "--family2",
            "-o",
            "eclipse/include/props/relperm-fam2.inc",
        ],
    )
    pyscalcli.main()
    assert Path("eclipse/include/props/relperm-fam2.inc").is_file()
    assert not any(record.levelno == logging.ERROR for record in caplog.records)

    caplog.clear()
    mocker.patch(
        "sys.argv", ["pyscal", str(relperm_file), "-o", "include/props/relperm.inc"]
    )
    pyscalcli.main()
    assert Path("include/props/relperm.inc").is_file()
    assert not any(record.levelno == logging.ERROR for record in caplog.records)

    caplog.clear()
    # Check that we can read specific sheets
    mocker.patch(
        "sys.argv",
        [
            "pyscal",
            str(relperm_file),
            "--sheet_name",
            "relperm",
            "--output",
            "relperm-firstsheet.inc",
        ],
    )
    pyscalcli.main()
    assert not any(record.levelno == logging.ERROR for record in caplog.records)

    # Identical files:
    assert len(
        Path("relperm-firstsheet.inc").read_text(encoding="utf8").splitlines()
    ) == len(Path("relperm.inc").read_text(encoding="utf8").splitlines())

    # Check that we can read specific sheets
    caplog.clear()
    mocker.patch(
        "sys.argv",
        [
            "pyscal",
            str(relperm_file),
            "--sheet_name",
            "simple",
            "--output",
            "relperm-secondsheet.inc",
        ],
    )
    pyscalcli.main()
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    secondsheet = os.linesep.join(
        Path("relperm-secondsheet.inc").read_text(encoding="utf8").splitlines()
    )
    assert "SATNUM 3" not in secondsheet
    assert "sand" in secondsheet
    assert "mud" in secondsheet  # From the comment column in sheet: simple

    # Check that we can read specific sheets
    caplog.clear()
    mocker.patch(
        "sys.argv",
        [
            "pyscal",
            str(relperm_file),
            "--sheet_name",
            "NOTEXISTINGÆÅ",
            "--output",
            "relperm-empty.inc",
        ],
    )
    with pytest.raises(SystemExit):
        pyscalcli.main()
    assert not Path("relperm-empty.inc").is_file()

    caplog.clear()
    mocker.patch(
        "sys.argv",
        ["pyscal", str(relperm_file), "--delta_s", "0.1", "-o", "deltas0p1.inc"],
    )
    pyscalcli.main()
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    linecount1 = len(Path("deltas0p1.inc").read_text(encoding="utf8").splitlines())

    caplog.clear()
    mocker.patch(
        "sys.argv",
        [
            "pyscal",
            str(relperm_file),
            "--delta_s",
            "0.01",
            "-o",
            "deltas0p01.inc",
        ],
    )
    pyscalcli.main()
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    linecount2 = len(Path("deltas0p01.inc").read_text(encoding="utf8").splitlines())
    assert linecount2 > linecount1 * 4  # since we don't filter out non-numerical lines


def test_pyscalcli_stdout_output(capsys, mocker):
    """Test that we can write to stdout"""
    scalrec_file = Path(__file__).absolute().parent / "data/scal-pc-input-example.xlsx"
    mocker.patch(
        "sys.argv",
        ["pyscal", str(scalrec_file), "--int_param_wo", "0", "--output", "-"],
    )
    pyscalcli.main()
    captured = capsys.readouterr()
    assert "SWOF" in captured.out

    mocker.patch(
        "sys.argv",
        [
            "pyscal",
            str(scalrec_file),
            "--family2",
            "--int_param_wo",
            "0",
            "--output",
            "-",
        ],
    )
    pyscalcli.main()
    captured = capsys.readouterr()
    assert "SOF3" in captured.out


def test_pyscalcli_exception_catching(capsys, mocker):
    """The command line client catches selected exceptions.

    Traceback is always included."""
    mocker.patch("sys.argv", ["pyscal", "notexisting.xlsx"])
    with pytest.raises(SystemExit, match="File not found"):
        pyscalcli.main()
    outerr = capsys.readouterr().out + capsys.readouterr().err
    assert "raise" in outerr  # This is the traceback.


@pytest.mark.skipif(sys.platform == "win32", reason="UTF-8 problems on Windows")
def test_pyscalcli_oilwater(tmp_path, caplog, mocker):
    """Test the command line client in two-phase oil-water"""
    os.chdir(tmp_path)
    relperm_file = "oilwater.csv"
    pd.DataFrame(
        # "fooå" here causes problems on Windows
        columns=["SATNUM", "nw", "now", "tag"],
        data=[[1, 2, 3, "fooå"]],
    ).to_csv(relperm_file, index=False)
    caplog.clear()
    mocker.patch(
        "sys.argv",
        [
            "pyscal",
            relperm_file,
            "--output",
            "ow.inc",
        ],
    )
    pyscalcli.main()
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    lines = Path("ow.inc").read_text(encoding="utf8").splitlines()
    joined = os.linesep.join(lines)
    assert "fooå" in joined
    assert 100 < len(lines) < 120  # weak test..

    # Test with SCAL recommendation:
    pd.DataFrame(
        columns=["SATNUM", "case", "nw", "now", "tag"],
        data=[
            [1, "low", 2, 3, "fooå"],
            [1, "base", 2, 3, "fooå"],
            [1, "high", 2, 3, "fooå"],
        ],
    ).to_csv(relperm_file, index=False)
    caplog.clear()
    mocker.patch(
        "sys.argv",
        [
            "pyscal",
            relperm_file,
            "--int_param_wo",
            "-0.1",
            "--output",
            "ow-int.inc",
        ],
    )
    pyscalcli.main()
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    assert not any(record.levelno == logging.ERROR for record in caplog.records)


def test_pyscalcli_gaswater(tmp_path, caplog, mocker):
    """Test the command line endpoint on gas-water problems"""
    os.chdir(tmp_path)
    relperm_file = "gaswater.csv"
    pd.DataFrame(columns=["SATNUM", "nw", "ng"], data=[[1, 2, 3]]).to_csv(
        relperm_file, index=False
    )
    caplog.clear()
    mocker.patch(
        "sys.argv",
        [
            "pyscal",
            relperm_file,
            "--output",
            "gw.inc",
        ],
    )
    pyscalcli.main()
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    lines = Path("gw.inc").read_text(encoding="utf8").splitlines()
    joined = os.linesep.join(lines)
    assert "SWFN" in joined
    assert "SGFN" in joined
    assert "SWOF" not in joined
    assert "sgrw" in joined
    assert "krgendanchor" not in joined
    assert "sorw" not in joined
    assert "sorg" not in joined
    assert len(lines) > 40


def test_pyscalcli_gaswater_scal(tmp_path, caplog, mocker):
    """Test the command line endpoint on gas-water problems, with
    interpolation"""
    os.chdir(tmp_path)
    relperm_file = "gaswater.csv"
    pd.DataFrame(
        columns=["SATNUM", "CASE", "nw", "ng"],
        data=[[1, "pess", 2, 3], [1, "base", 3, 4], [1, "opt", 5, 6]],
    ).to_csv(relperm_file, index=False)

    caplog.clear()
    mocker.patch(
        "sys.argv",
        [
            "pyscal",
            relperm_file,
            "--int_param_wo",
            "-0.2",
            "--output",
            "gw.inc",
        ],
    )
    pyscalcli.main()
    assert not any(record.levelno == logging.INFO for record in caplog.records)
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    lines = Path("gw.inc").read_text(encoding="utf8").splitlines()
    joined = os.linesep.join(lines)
    assert "SWFN" in joined
    assert "SGFN" in joined
    assert "SWOF" not in joined
    assert "sgrw" in joined
    assert "krgendanchor" not in joined
    assert "sorw" not in joined
    assert "sorg" not in joined
    assert len(lines) > 40


def test_pyscal_client_scal(tmp_path, caplog, default_loglevel, mocker):
    # pylint: disable=unused-argument
    # default_loglevel fixture is in conftest.py
    """Test the command line endpoint on SCAL recommendation"""
    scalrec_file = Path(__file__).absolute().parent / "data/scal-pc-input-example.xlsx"

    os.chdir(tmp_path)

    mocker.patch("sys.argv", ["pyscal", str(scalrec_file)])
    with pytest.raises(SystemExit):
        pyscalcli.main()

    caplog.clear()
    mocker.patch(
        "sys.argv",
        ["pyscal", str(scalrec_file), "--int_param_wo", 0, "-o", "relperm1.inc"],
    )
    pyscalcli.main()
    assert not any(record.levelno == logging.INFO for record in caplog.records)
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    assert not any(record.levelno == logging.ERROR for record in caplog.records)

    relpermlines = os.linesep.join(
        Path("relperm1.inc").read_text(encoding="utf8").splitlines()
    )
    assert "SWOF" in relpermlines
    assert "SGOF" in relpermlines
    assert "SLGOF" not in relpermlines
    assert "SOF3" not in relpermlines
    sat_table_str_ok(relpermlines)
    # assert "int_param_wo: 0\n" in relpermlines  # this should be in the tag.

    caplog.clear()
    mocker.patch(
        "sys.argv",
        [
            "pyscal",
            str(scalrec_file),
            "--int_param_wo",
            "-0.5",
            "-o",
            "relperm2.inc",
        ],
    )
    pyscalcli.main()
    assert not any(record.levelno == logging.INFO for record in caplog.records)
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    # assert something about -0.5 in the comments


def test_pyscal_client_error(tmp_path, mocker):
    """Test various error conditions, asserting the the correct error message is emitted

    Some error are caught in pyscalcli.py, some errors are caught when loading the xlsx
    file"""

    os.chdir(tmp_path)
    scalrec_file = Path(__file__).absolute().parent / "data/scal-pc-input-example.xlsx"

    # int_param_go should not be used alone:
    mocker.patch("sys.argv", ["pyscal", str(scalrec_file), "--int_param_go", "-0.5"])
    with pytest.raises(SystemExit, match="Don't use int_param_go alone"):
        pyscalcli.main()

    # Delete SATNUM from xlsx input:
    pd.read_excel(scalrec_file, engine="openpyxl").drop("SATNUM", axis=1).to_csv(
        "no_satnum.csv"
    )
    mocker.patch("sys.argv", ["pyscal", "no_satnum.csv", "--int_param_wo", "-0.5"])
    with pytest.raises(SystemExit, match="SATNUM must be present"):
        pyscalcli.main()

    # Delete CASE from xlsx input:
    pd.read_excel(scalrec_file, engine="openpyxl").drop("CASE", axis=1).to_csv(
        "no_case.csv"
    )
    mocker.patch("sys.argv", ["pyscal", "no_case.csv", "--int_param_wo", "-0.5"])
    with pytest.raises(SystemExit, match="Non-unique SATNUMs"):
        pyscalcli.main()

    # Multiple interpolation parameters, this was supported in pyscal <= 0.7.7,
    # but is now an error (raised by argparse):
    mocker.patch(
        "sys.argv", ["pyscal", str(scalrec_file), "--int_param_wo", "-0.5", "0"]
    )
    with pytest.raises(SystemExit):
        pyscalcli.main()


@pytest.mark.parametrize(
    "int_param_wo, int_param_go, raises, match",
    [
        (-1, None, None, None),
        (-2, None, ValueError, "must be in"),
        (-1, -2, ValueError, "must be in"),
        (None, -1, ValueError, "int_param_go alone"),
        (None, None, ValueError, None),
        ([], None, TypeError, "SATNUM specific interpolation"),
        ([-1], None, TypeError, "SATNUM specific interpolation"),
        ([1, 1], None, TypeError, "SATNUM specific interpolation"),
        ([], [], TypeError, "SATNUM specific interpolation"),
        (-1, [], TypeError, "SATNUM specific interpolation"),
        (None, [], ValueError, "int_param_go alone"),
    ],
)
def test_pyscal_main_interpolation_parameters(
    int_param_wo, int_param_go, raises, match
):
    """Define the behaviour on different interpolation parameter combinations.

    Earlier pyscal versions allowed lists of parameters as input.

    The command line client will also catch these errors through argparse, but
    when pyscal is used e.g. in fm_pyscal.py in semeio, these errors need to
    be caught by the main() function.
    """
    scalrec_file = Path(__file__).absolute().parent / "data/scal-pc-input-example.xlsx"
    if raises is not None:
        with pytest.raises(raises, match=match):
            pyscalcli.pyscal_main(
                scalrec_file,
                int_param_wo=int_param_wo,
                int_param_go=int_param_go,
                output=os.devnull,
            )
    else:
        pyscalcli.pyscal_main(
            scalrec_file,
            int_param_wo=int_param_wo,
            int_param_go=int_param_go,
            output=os.devnull,
        )


def test_pyscal_main():
    """The pyscal client is a wrapper main() function that runs argparse, and then
    hands over responsibility to pyscal_main(). This wrapping is to facilitate
    fm_pyscal.py in semeio f.ex.

    This test function is for testing e.g error scenarios that argparse
    would catch, but that we also need to check on behalf of semeio usage."""

    # This input file has no CASE column, and interpolation is not meaningful.
    relperm_file = Path(__file__).absolute().parent / "data/relperm-input-example.xlsx"

    pyscalcli.pyscal_main(relperm_file, output=os.devnull)

    with pytest.raises(ValueError, match="Interpolation parameter provided"):
        pyscalcli.pyscal_main(relperm_file, int_param_wo=-1, output=os.devnull)
