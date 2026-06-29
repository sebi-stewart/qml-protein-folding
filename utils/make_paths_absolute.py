"""
Ensure the current working directory is set to the repository root so relative paths are stable.
This module changes the process CWD on import to make notebooks and scripts path-independent.
"""

from os import chdir
from pathlib import Path


def make_paths_relative_to_root():
    """Always use the same, absolute (relative to root) paths

    which makes moving the notebooks around easier.
    """
    top_level = Path(__file__).parent.parent

    chdir(top_level)


make_paths_relative_to_root()