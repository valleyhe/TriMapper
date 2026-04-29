"""Lazy import utility for optional dependencies.

Provides require() function that imports packages with helpful error messages
if the package is not installed.
"""

from __future__ import annotations


def require(pkg_name: str, purpose: str):
    """Import package, raising ImportError with context if missing.

    Parameters
    ----------
    pkg_name : str
        Name of the package to import (e.g., "ot", "gseapy")
    purpose : str
        Description of what the package is needed for (e.g., "optimal transport")

    Returns
    -------
    module
        The imported module

    Raises
    ------
    ImportError
        If the package is not installed, with a helpful message including
        installation instructions.

    Examples
    --------
    >>> ot = require("ot", "optimal transport")
    >>> gseapy = require("gseapy", "ssGSEA bulk preprocessing")
    """
    try:
        return __import__(pkg_name)
    except ImportError:
        raise ImportError(
            f"Package '{pkg_name}' required for {purpose}. "
            f"Install with: pip install {pkg_name}"
        )