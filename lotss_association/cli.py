"""Installed command-line entry points for LoTSS Association.

The historical ``scripts/`` modules remain available for users who invoke
them directly.  This module provides equivalent installed commands so a
wheel installation does not depend on the repository working directory.
"""

from __future__ import annotations

import sys
from collections.abc import Callable


def _dispatch(module_main: Callable[[], None], argv: list[str]) -> None:
    original = sys.argv
    try:
        sys.argv = [original[0], *argv]
        module_main()
    finally:
        sys.argv = original


def main() -> None:
    """Run the Stage 1 pipeline (``lotss-association``)."""

    from scripts.run_pipeline import main as pipeline_main

    _dispatch(pipeline_main, sys.argv[1:])


def parent_link_main() -> None:
    """Run Stage 2 parent linking (``lotss-parent-link``)."""

    from scripts.run_parent_linking import main as parent_main

    _dispatch(parent_main, sys.argv[1:])


def visualize_main() -> None:
    """Run diagnostic visualization (``lotss-visualize``)."""

    from scripts.visualize_results import main as visualize

    _dispatch(visualize, sys.argv[1:])
