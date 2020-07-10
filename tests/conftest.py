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
