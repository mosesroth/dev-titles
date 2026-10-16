# Development

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,cjk]'          # add [whisper] / [parakeet] for local ASR,
                                     # [api] / [bedrock] for SDK translators, [youtube] for publishing
```

The code lives under `src/dev_titles/` and the tests under `test/`.

## Testing

```bash
pytest                                   # whole suite (config in setup.cfg [tool:pytest])
pytest test/test_resegment.py -q         # one area
pytest --cov dev_titles --cov-report term-missing   # with coverage (pytest-cov, in [dev])
```

The unit tests are meant to run offline. LLM steps are exercised with fake translators
and stub `invoke` callables (see `test/helpers.py`).

## Linting and formatting

```bash
black src test && isort src test     # formatting (line length 100, see pyproject.toml)
flake8 src test                      # config in setup.cfg [flake8]
mypy src                             # config in pyproject.toml [tool.mypy]
```

## Running locally

```bash
dev-titles --help                  # console script installed by pip install -e .
python -m dev_titles.cli --help    # equivalent, without installing the script
```

## Docs (optional)

The API reference is built with Sphinx and MyST:

```bash
pip install sphinx myst-parser
sphinx-build -b html doc build/doc
```
