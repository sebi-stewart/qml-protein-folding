"""
Project-wide constants governing QAOA defaults and environment-derived adjustments.
Defines sensible defaults for layers, optimisation and platform-dependent fallbacks.
"""

import sys

QAOA_LAYERS = 8

IS_LINUX = sys.platform == "linux"

if not IS_LINUX:
    QAOA_LAYERS = 2
