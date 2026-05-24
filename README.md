# griit

A Python package with a `griit` command-line interface.

## Installation

Install in editable mode for development:

```bash
pip install -e ".[dev]"
```

## Usage

```bash
griit --help
griit --version
```

## Project layout

```
griit/
├── pyproject.toml
├── README.md
├── src/
│   └── griit/
│       ├── __init__.py
│       └── cli.py
└── tests/
    └── test_cli.py
```

## Development

Run the test suite with:

```bash
pytest
```
