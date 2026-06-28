# Wildfire ML project commands

install:
    uv sync --all-groups
    uv run pre-commit install

format:
    uv run black src tests
    uv run ruff check --fix src tests

lint:
    uv run ruff check src tests
    uv run black --check src tests

test:
    uv run pytest

train:
    uv run python -m wildfire.train
