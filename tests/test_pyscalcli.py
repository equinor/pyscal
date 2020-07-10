"""Test the pyscal client"""
# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import logging

import pandas as pd

import pytest
import subprocess

from pyscal import pyscalcli
from common import sat_table_str_ok


@pytest.mark.integration
def test_installed():
    """Test that the command line client is installed in PATH and
    starts up nicely"""
    assert subprocess.check_output(["pyscal", "-h"])


def test_log_levels(tmpdir, caplog, default_loglevel):
    """Test that we can control the log level from the command line
    client, and get log output from modules deep down"""
    if "__file__" in globals():
        # Easen up copying test code into interactive sessions
        testdir = os.path.dirname(os.path.abspath(__file__))
    else:
        testdir = os.path.abspath(".")

    relperm_file = testdir + "/data/relperm-input-example.xlsx"

    tmpdir.chdir()

    caplog.clear()
    sys.argv = ["pyscal", relperm_file]
    pyscalcli.main()

    # Ensure we have no INFO logging:
    assert not any(record.levelno == logging.INFO for record in caplog.records)

    # The following is an INFO statement from factory.py that we should not get:
    assert all("Loaded input data" not in str(record) for record in caplog.records)
    # And a debug statement from gasoil.py
    assert all(
        "Added Corey gas to GasOil object" not in str(record)
        for record in caplog.records
    )

    caplog.clear()
    sys.argv = ["pyscal", "--verbose", relperm_file]
    pyscalcli.main()
    assert not any(record.levelno == logging.DEBUG for record in caplog.records)
    assert any(record.levelno == logging.INFO for record in caplog.records)
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    assert any("Loaded input data" in str(record) for record in caplog.records)
    assert any("Dumping" in str(record) for record in caplog.records)

    caplog.clear()
    sys.argv = ["pyscal", "--debug", relperm_file]
    pyscalcli.main()
    assert any(record.levelno == logging.DEBUG for record in caplog.records)
    assert any("Initialized GasOil with" in str(record) for record in caplog.records)
    assert any("Initialized WaterOil with" in str(record) for record in caplog.records)


def test_pyscal_client_static(tmpdir, caplog, default_loglevel):
    """Test pyscal client for static relperm input"""
    if "__file__" in globals():
        # Easen up copying test code into interactive sessions
        testdir = os.path.dirname(os.path.abspath(__file__))
    else:
        testdir = os.path.abspath(".")

    relperm_file = testdir + "/data/relperm-input-example.xlsx"

    tmpdir.chdir()

    caplog.clear()
    sys.argv = ["pyscal", relperm_file]
    pyscalcli.main()
    assert os.path.exists("relperm.inc")
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    assert not any(record.levelno == logging.INFO for record in caplog.records)

    # We get one warning due to empty cells in xlsx:
    assert sum(record.levelno == logging.WARNING for record in caplog.records) == 1

    relpermlines = "\n".join(open("relperm.inc").readlines())
    assert "SWOF" in relpermlines
    assert "SGOF" in relpermlines
    assert "SLGOF" not in relpermlines
    assert "SOF3" not in relpermlines
    sat_table_str_ok(relpermlines)

    caplog.clear()
    sys.argv = ["pyscal", relperm_file, "--output", "alt2relperm.inc"]
    pyscalcli.main()
    assert os.path.exists("alt2relperm.inc")
    assert not any(record.levelno == logging.ERROR for record in caplog.records)

    caplog.clear()
    sys.argv = ["pyscal", relperm_file, "-o", "altrelperm.inc"]
    pyscalcli.main()
    assert os.path.exists("altrelperm.inc")
    assert not any(record.levelno == logging.ERROR for record in caplog.records)

    caplog.clear()
    sys.argv = ["pyscal", relperm_file, "--family2", "-o", "relperm-fam2.inc"]
    pyscalcli.main()
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    assert os.path.exists("relperm-fam2.inc")
    relpermlines = "\n".join(open("relperm-fam2.inc").readlines())
    assert "SWFN" in relpermlines
    assert "SGFN" in relpermlines
    assert "SOF3" in relpermlines
    assert "SWOF" not in relpermlines
    assert "SGOF" not in relpermlines
    sat_table_str_ok(relpermlines)

    caplog.clear()
    sys.argv = ["pyscal", relperm_file, "--slgof", "--output", "relperm-slgof.inc"]
    pyscalcli.main()
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    assert os.path.exists("relperm-slgof.inc")
    relpermlines = "\n".join(open("relperm-slgof.inc").readlines())
    assert "SWOF" in relpermlines
    assert "SGOF" not in relpermlines
    assert "SLGOF" in relpermlines
    assert "SOF3" not in relpermlines
    sat_table_str_ok(relpermlines)

    caplog.clear()
    # Dump to deep directory structure that does not exists
    sys.argv = [
        "pyscal",
        relperm_file,
        "--family2",
        "-o",
        "eclipse/include/props/relperm-fam2.inc",
    ]
    pyscalcli.main()
    assert os.path.exists("eclipse/include/props/relperm-fam2.inc")
    assert not any(record.levelno == logging.ERROR for record in caplog.records)

    caplog.clear()
    sys.argv = ["pyscal", relperm_file, "-o", "include/props/relperm.inc"]
    pyscalcli.main()
    assert os.path.exists("include/props/relperm.inc")
    assert not any(record.levelno == logging.ERROR for record in caplog.records)

    caplog.clear()
    # Check that we can read specific sheets
    sys.argv = [
        "pyscal",
        relperm_file,
        "--sheet_name",
        "relperm",
        "--output",
        "relperm-firstsheet.inc",
    ]
    pyscalcli.main()
    assert not any(record.levelno == logging.ERROR for record in caplog.records)

    # Identical files:
    assert len(open("relperm-firstsheet.inc").readlines()) == len(
        open("relperm.inc").readlines()
    )

    # Check that we can read specific sheets
    caplog.clear()
    sys.argv = [
        "pyscal",
        relperm_file,
        "--sheet_name",
        "simple",
        "--output",
        "relperm-secondsheet.inc",
    ]
    pyscalcli.main()
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    secondsheet = "\n".join(open("relperm-secondsheet.inc").readlines())
    assert "SATNUM 3" not in secondsheet
    assert "sand" in secondsheet
    assert "mud" in secondsheet  # From the comment column in sheet: simple

    # Check that we can read specific sheets
    caplog.clear()
    sys.argv = [
        "pyscal",
        relperm_file,
        "--sheet_name",
        u"NOTEXISTINGÆÅ",
        "--output",
        "relperm-empty.inc",
    ]
    with pytest.raises(SystemExit):
        pyscalcli.main()
    assert not os.path.exists("relperm-empty.inc")

    caplog.clear()
    sys.argv = ["pyscal", relperm_file, "--delta_s", "0.1", "-o", "deltas0p1.inc"]
    pyscalcli.main()
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    linecount1 = len(open("deltas0p1.inc").readlines())

    caplog.clear()
    sys.argv = ["pyscal", relperm_file, "--delta_s", "0.01", "-o", "deltas0p01.inc"]
    pyscalcli.main()
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    linecount2 = len(open("deltas0p01.inc").readlines())
    assert linecount2 > linecount1 * 4  # since we don't filter out non-numerical lines


def test_pyscalcli_oilwater(tmpdir, caplog, default_loglevel):
    """Test the command line client in two-phase oil-water"""
    tmpdir.chdir()
    relperm_file = "oilwater.csv"
    pd.DataFrame(
        columns=["SATNUM", "nw", "now", "tag"], data=[[1, 2, 3, "fooå"]]
    ).to_csv(relperm_file, index=False)
    caplog.clear()
    sys.argv = [
        "pyscal",
        relperm_file,
        "--output",
        "ow.inc",
    ]
    pyscalcli.main()
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    lines = open("ow.inc").readlines()
    joined = "\n".join(lines)
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
    sys.argv = [
        "pyscal",
        relperm_file,
        "--int_param_wo",
        "-0.1",
        "--output",
        "ow-int.inc",
    ]
    pyscalcli.main()
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    assert not any(record.levelno == logging.ERROR for record in caplog.records)


def test_pyscalcli_gaswater(tmpdir, caplog, default_loglevel):
    """Test the command line endpoint on gas-water problems"""
    tmpdir.chdir()
    relperm_file = "gaswater.csv"
    pd.DataFrame(columns=["SATNUM", "nw", "ng"], data=[[1, 2, 3]]).to_csv(
        relperm_file, index=False
    )
    caplog.clear()
    sys.argv = [
        "pyscal",
        relperm_file,
        "--output",
        "gw.inc",
    ]
    pyscalcli.main()
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    lines = open("gw.inc").readlines()
    joined = "\n".join(lines)
    assert "SWFN" in joined
    assert "SGFN" in joined
    assert "SWOF" not in joined
    assert "sgrw" in joined
    assert "krgendanchor" not in joined
    assert "sorw" not in joined
    assert "sorg" not in joined
    assert len(lines) > 40


def test_pyscalcli_gaswater_scal(tmpdir, caplog, default_loglevel):
    """Test the command line endpoint on gas-water problems, with
    interpolation"""
    tmpdir.chdir()
    relperm_file = "gaswater.csv"
    pd.DataFrame(
        columns=["SATNUM", "CASE", "nw", "ng"],
        data=[[1, "pess", 2, 3], [1, "base", 3, 4], [1, "opt", 5, 6]],
    ).to_csv(relperm_file, index=False)

    caplog.clear()
    sys.argv = [
        "pyscal",
        relperm_file,
        "--int_param_wo",
        "-0.2",
        "--output",
        "gw.inc",
    ]
    pyscalcli.main()
    assert not any(record.levelno == logging.INFO for record in caplog.records)
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    lines = open("gw.inc").readlines()
    joined = "\n".join(lines)
    assert "SWFN" in joined
    assert "SGFN" in joined
    assert "SWOF" not in joined
    assert "sgrw" in joined
    assert "krgendanchor" not in joined
    assert "sorw" not in joined
    assert "sorg" not in joined
    assert len(lines) > 40


def test_pyscal_client_scal(tmpdir, caplog, default_loglevel):
    """Test the command line endpoint on SCAL recommendation"""
    if "__file__" in globals():
        # Easen up copying test code into interactive sessions
        testdir = os.path.dirname(os.path.abspath(__file__))
    else:
        testdir = os.path.abspath(".")

    scalrec_file = testdir + "/data/scal-pc-input-example.xlsx"

    tmpdir.chdir()

    sys.argv = ["pyscal", scalrec_file]
    with pytest.raises(SystemExit):
        pyscalcli.main()

    caplog.clear()
    sys.argv = ["pyscal", scalrec_file, "--int_param_wo", 0, "-o", "relperm1.inc"]
    pyscalcli.main()
    assert not any(record.levelno == logging.INFO for record in caplog.records)
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    assert not any(record.levelno == logging.ERROR for record in caplog.records)

    relpermlines = "\n".join(open("relperm1.inc").readlines())
    assert "SWOF" in relpermlines
    assert "SGOF" in relpermlines
    assert "SLGOF" not in relpermlines
    assert "SOF3" not in relpermlines
    sat_table_str_ok(relpermlines)
    # assert "int_param_wo: 0\n" in relpermlines  # this should be in the tag.

    caplog.clear()
    sys.argv = ["pyscal", scalrec_file, "--int_param_wo", "-0.5", "-o", "relperm2.inc"]
    pyscalcli.main()
    assert not any(record.levelno == logging.INFO for record in caplog.records)
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    # assert something about -0.5 in the comments

    # Only two interpolation parameters for three satnums:
    sys.argv = ["pyscal", scalrec_file, "--int_param_wo", "-0.5", "0"]
    with pytest.raises(SystemExit):
        pyscalcli.main()

    caplog.clear()
    sys.argv = [
        "pyscal",
        scalrec_file,
        "--int_param_wo",
        "-0.5",
        "0.0",
        "1.0",
        "-o",
        "relperm3.inc",
    ]
    pyscalcli.main()
    assert not any(record.levelno == logging.INFO for record in caplog.records)
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    assert not any(record.levelno == logging.ERROR for record in caplog.records)
    assert os.path.exists("relperm3.inc")
    # assert someting about three different parameters..

    sys.argv = [
        "pyscal",
        scalrec_file,
        "--int_param_wo",
        "-0.5",
        "0",
        "1",
        "--int_param_go",
        "0.9",
        "-o",
        "relperm4.inc",
    ]
    pyscalcli.main()
    assert os.path.exists("relperm4.inc")

    sys.argv = [
        "pyscal",
        scalrec_file,
        "--int_param_wo",
        "-0.5",
        "0",
        "1",
        "--int_param_go",
        "0.9",
        "1",
    ]
    with pytest.raises(SystemExit):
        pyscalcli.main()
    sys.argv = [
        "pyscal",
        scalrec_file,
        "--int_param_wo",
        "-0.5",
        "0",
        "1",
        "--int_param_go",
        "0.9",
        "1",
        "-0.5",
        "-o",
        "relperm5.inc",
    ]
    pyscalcli.main()
    assert os.path.exists("relperm5.inc")
    # check that interpolation parameters have been used.
