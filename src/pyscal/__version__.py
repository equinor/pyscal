try:
    from .version import version

    __version__ = version
except ImportError:
    __version__ = "0.0.0"
