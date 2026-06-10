"""Setuptools compatibility shim.

Package metadata lives in pyproject.toml. This file exists only for legacy tools
that still expect setup.py.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
