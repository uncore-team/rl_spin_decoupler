# Contributing

Thank you for contributing to RL Spin Decoupler.

## Development setup

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/uncore-team/rl_spin_decoupler.git
cd rl_spin_decoupler
python -m pip install -e ".[dev]"
python -m pip install -e ".[docs]"
```

Optional: install pre-commit hooks so linting and formatting run before each commit.

```bash
pre-commit install
```

## Test commands

Run the complete test suite:

```bash
pytest
```

Generate coverage artifacts:

```bash
pytest --cov-report=xml --cov-report=html
```

## Linting and formatting

The repository uses Ruff for import sorting, linting, and formatting.

Check code style:

```bash
ruff check .
ruff format --check .
```

Apply formatting automatically:

```bash
ruff format .
```

## Documentation

Build the Sphinx documentation locally:

```bash
python -m sphinx -b html docs docs/_build/html
```

The output home page is:

```text
docs/_build/html/index.html
```

## Pull requests

- Keep changes focused and easy to review.
- Update tests when changing behaviour.
- Update documentation when changing public APIs or workflows.
- Prefer opening pull requests from a topic branch rather than committing directly to `main`.