"""This file gets auto-discovered by pytest, and is the recommended
place to have common fixtures, and avoiding flake8 complaints"""

import logging

import pytest


@pytest.fixture
def default_loglevel():
    """Reset the log level for all registered loggers to WARNING

    This is necessary when testing for appearance of different log
    messages, when some of the tested commands might manipulate the log
    levels in order to set INFO or DEBUG messages, e.g. through --verbose
    sent to the command line client"""
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.WARNING)
    return default_loglevel


def pytest_addoption(parser):
    parser.addoption(
        "--plot",
        action="store_true",
        default=False,
        help="Run tests that display plots to the screen",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--plot"):
        # Do not skip tests when --plot is supplied on pytest command line
        return
    skip_plot = pytest.mark.skip(reason="need --plot option to run")
    for item in items:
        if "plot" in item.keywords:
            item.add_marker(skip_plot)
