"""Command-line entry point for griit."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the griit CLI."""
    parser = argparse.ArgumentParser(
        prog="griit",
        description="Griit command-line interface.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the griit CLI.

    Returns the process exit code.
    """
    parser = build_parser()
    parser.parse_args(argv)
    print("griit: nothing to do yet")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
