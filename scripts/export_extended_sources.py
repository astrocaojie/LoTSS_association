#!/usr/bin/env python
"""Export merged source catalogs by running the pipeline entry point."""

from __future__ import annotations

try:
    from .run_pipeline import main
except ImportError:  # pragma: no cover - direct script compatibility
    from run_pipeline import main

if __name__ == "__main__":
    main()
