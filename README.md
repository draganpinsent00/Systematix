[![CI](https://github.com/your-org/your-repo/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/your-repo/actions/workflows/ci.yml)

# Systematix — Monte Carlo Options Pricing Dashboard (MVP)

A brief description of your project.

## Installation

Instructions on how to install the project.

## Usage

Instructions on how to use the project.

## Continuous Integration

A GitHub Actions workflow is included at `.github/workflows/ci.yml` which installs dependencies and runs pytest on push and PRs to `main`/`master`.

## Developer quick commands

Run unit tests:

```powershell
py -3 -m pytest -q
```

Run the smoke test:

```powershell
py -3 -u .\smoke_test.py
```

Run the Streamlit dashboard:

```powershell
py -3 -m streamlit run dashboard.py
```

If you are using `python` directly instead of the `py` launcher, substitute `python` for `py -3`.
