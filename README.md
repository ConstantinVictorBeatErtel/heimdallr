# Wildfire

Next-day wildfire spread forecasting.

This project predicts how active wildfires are likely to spread over the following 24 hours using geospatial, weather, fuel, and terrain features.

## Getting started

1. Install [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).
2. Run `just install` to create the virtual environment and install dependencies.
3. Run `just test` to verify the setup.

## Common commands

| Command        | Description                          |
| -------------- | ------------------------------------ |
| `just install` | Install dependencies and pre-commit  |
| `just format`  | Auto-format code with Black and Ruff |
| `just lint`    | Check formatting and lint rules      |
| `just test`    | Run the test suite                   |
| `just train`   | Run the training entry point         |

## Project layout

```
src/wildfire/   Application code
tests/          Pytest tests
```

## Status

Early scaffold — data ingestion, modeling, and evaluation pipelines are not implemented yet.
