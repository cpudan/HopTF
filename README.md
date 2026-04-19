# HopTF

This repository now uses two separate `uv` environments:

- The root project is the main environment. It includes `scarf` plus the dependencies needed for local development and tests.
- [`envs/alphagenome/pyproject.toml`](/home/dmeyer/courses/clmm/HopTF/envs/alphagenome/pyproject.toml) is a separate `uv` project for the much heavier AlphaGenome stack.

Separating them avoids dependency conflicts between SCARF and AlphaGenome/TensorFlow while keeping the default environment much smaller.

## Install `uv`

If `uv` is not already installed, one simple option is:

```bash
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
```

## Set Up The Main Environment

From the repository root:

```bash
uv sync
```

This creates `.venv/` from the root [`uv.lock`](./uv.lock).

Note: with Python `3.13`, SCARF's pinned `numpy==1.26.4` may build from source the first time. If you want faster wheel-based setup, Python `3.12` is the smoother option for the main environment.

## Run Tests

The test suite lives under [`tests/`](./tests).

Run it with:

```bash
uv run pytest
```

For a quieter success-only run:

```bash
uv run pytest -q
```

## Set Up The AlphaGenome Environment

The AlphaGenome dependencies are intentionally isolated in their own `uv` project:

```bash
cd envs/alphagenome
uv sync
```

This creates `envs/alphagenome/.venv/` from [`envs/alphagenome/uv.lock`](./envs/alphagenome/uv.lock).

Use this environment when working with AlphaGenome-specific scripts or notebooks. Keep using the root environment for SCARF work and for running the repository tests.
