from pathlib import Path

import pytest

import pyscal
from pyscal import pyscalcli

try:
    import opm  # noqa

    HAVE_OPM = True
except ImportError:
    HAVE_OPM = False


def test_default_logger_levels_and_split(capsys):
    """Verify that the intended usage of this logger have expected results"""

    splitlogger = pyscal.getLogger_pyscal("test_levels_split")

    splitlogger.debug("This DEBUG-text is not to be seen")
    captured = capsys.readouterr()
    assert "DEBUG-text" not in captured.out
    assert "DEBUG-text" not in captured.err

    splitlogger.info("This INFO-text is not to be seen by default")
    captured = capsys.readouterr()
    assert "INFO-text" not in captured.out

    splitlogger.warning("This WARNING-text is to be seen in stdout")
    captured = capsys.readouterr()
    assert "WARNING-text" in captured.out
    assert "WARNING-text" not in captured.err

    splitlogger.error("This ERROR-text should only be in stderr")
    captured = capsys.readouterr()
    assert "ERROR-text" not in captured.out
    assert "ERROR-text" in captured.err

    # If output is written to stdout, all logs should go to stderr:
    nosplit_logger = pyscal.getLogger_pyscal(
        "test_levels_nosplit", args_dict={"output": "-", "debug": True}
    )
    nosplit_logger.debug("This DEBUG-text is to be seen in stderr")
    captured = capsys.readouterr()
    assert "DEBUG-text" not in captured.out
    assert "DEBUG-text" in captured.err

    nosplit_logger.info("This INFO-text is to be seen by in stderr")
    captured = capsys.readouterr()
    assert "INFO-text" not in captured.out
    assert "INFO-text" in captured.err

    nosplit_logger.warning("This WARNING-text is to be seen in stderr")
    captured = capsys.readouterr()
    assert "WARNING-text" not in captured.out
    assert "WARNING-text" in captured.err

    nosplit_logger.error("This ERROR-text should only be in stderr")
    captured = capsys.readouterr()
    assert "ERROR-text" not in captured.out
    assert "ERROR-text" in captured.err


@pytest.mark.skipif(not HAVE_OPM, reason="Command line client requires OPM")
def test_pyscal_logging_verbose(tmp_path, mocker, capsys):
    """Test that the command line client logs correctly with output set to
    stdout and verbose set to true.
    """

    testdir = Path(__file__).absolute().parent
    relperm_file = testdir / "data/relperm-input-example.xlsx"
    commands = ["pyscal", str(relperm_file), "--output"]
    commands.append("-")

    commands.append("-v")

    mocker.patch("sys.argv", commands)

    pyscalcli.main()
    captured = capsys.readouterr()
    stdout_output = captured.out
    stderr_output = captured.err

    assert "INFO:" in stderr_output
    assert "INFO:" not in stdout_output


@pytest.mark.skipif(not HAVE_OPM, reason="Command line client requires OPM")
def test_pyscal_logging(tmp_path, mocker, capsys):
    """Test that the command line client logs correctly with output set to
    stdout and verbose set to false.
    """

    testdir = Path(__file__).absolute().parent
    relperm_file = testdir / "data/relperm-input-example.xlsx"
    commands = ["pyscal", str(relperm_file), "--output"]
    commands.append("-")

    mocker.patch("sys.argv", commands)

    pyscalcli.main()
    captured = capsys.readouterr()
    stdout_output = captured.out
    stderr_output = captured.err

    assert "INFO:" not in stdout_output
    assert "INFO:" not in stderr_output


def test_repeated_logger_construction(capsys):
    """If we repeatedly call getLogger(), ensure handlers are not added on top"""
    logger = pyscal.getLogger_pyscal("nodouble")
    logger = pyscal.getLogger_pyscal("nodouble")
    logger.warning("Don't repeat me")
    captured = capsys.readouterr()
    assert captured.out.count("Don't repeat me") == 1
