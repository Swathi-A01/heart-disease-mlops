# Contributing

This repository was developed as part of BITS Pilani MTech AI/ML coursework (AIMLCZG523).

## Running Locally

```bash
git clone https://github.com/Swathi-A01/heart-disease-mlops.git
cd heart-disease-mlops
python -m venv venv && source venv/bin/activate
make install
make train
make serve
```

## Running Tests

```bash
make test    # runs all 25 pytest tests
make lint    # flake8 style check
```

## Project Structure

See [README.md](README.md) for the full file tree and architecture diagram.

## Code Style

- Python 3.11, flake8 enforced (max line length 100)
- All dependencies pinned in requirements.txt
- `random_state=42` on all stochastic operations for reproducibility

## Reporting Issues

Open a GitHub Issue with:
- A clear description of the problem
- The command that failed
- The full error output
