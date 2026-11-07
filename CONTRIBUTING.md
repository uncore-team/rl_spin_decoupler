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

## Writing style

Code comments and READMEs follow the same rules, so the project reads as if it
had one author. The rules for code also appear in `.github/copilot-instructions.md`.

Code and comments:

- Write everything in English, names included.
- Comment what the code cannot say: a reason, a constraint, a measured fact. Do
  not narrate what the next line does. A comment that records a measurement says
  what was measured and how big it was.
- Public functions and classes get Google-style docstrings (`Args`, `Returns`,
  `Raises`) with a one-sentence summary and 4-space continuation indents. Private
  helpers get a docstring only when the name is not enough.
- End-of-line comments are short lowercase fragments without a final period. Block
  comments are full sentences.
- No banners, boxes or separator lines.
- Shell scripts start with a header: what the script does, how to call it, and the
  environment variables it reads with their defaults.

READMEs:

- Start with the name of the thing as a title, then one or two paragraphs saying
  what it is and how it relates to the other examples. Next come, as far as they
  apply: Installation, Running (with Options and Timeouts), How it works, Files,
  Notes and References.
- Show commands from the repository root, with paths relative to it.
- Use a table only for parameters (name, default, meaning). Everything else is
  prose or a plain list.
- Write plainly. No bold for emphasis, no emoji, no marketing words ("robust",
  "seamless", "powerful"), and no dashes standing in for commas or parentheses.
  Give numbers, file names and commands rather than adjectives.

## Pull requests

- Keep changes focused and easy to review.
- Update tests when changing behaviour.
- Update documentation when changing public APIs or workflows.
- Prefer opening pull requests from a topic branch rather than committing directly to `main`.