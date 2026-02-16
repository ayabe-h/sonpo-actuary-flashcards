# Project Guidelines

## Code Style
- Python 3.10+ with Streamlit; keep UI orchestration in main module and move logic into helpers as specified in [docs/SPEC.md](docs/SPEC.md).
- Scoring must use difflib.SequenceMatcher with integer percent mapping per [docs/SPEC.md](docs/SPEC.md).

## Architecture
- Streamlit Q&A drill app for Japanese Actuary Second Exam; mobile-first usage. See [docs/SPEC.md](docs/SPEC.md).
- Enforce file boundaries: UI in main.py, scoring in scoring.py, selection in selector.py; no duplication. See [docs/SPEC.md](docs/SPEC.md).

## Build and Test
- No build/test commands documented in [README.md](README.md).
- Dependencies file exists as [requiements.txt](requiements.txt) but is empty.

## Project Conventions
- Data input is CSV with columns question,answer at data/cards.csv; progress tracking at data/progress.csv. See [docs/SPEC.md](docs/SPEC.md).
- progress.csv is auto-generated if missing; score_class is 0,1,2 with specific thresholds. See [docs/SPEC.md](docs/SPEC.md).

## Integration Points
- Local CSV files only; current data files are data/flashcards_*.csv.
- No external services, ML, or LLMs. See [docs/SPEC.md](docs/SPEC.md).

## Security
- No authentication or user accounts; no sensitive data handling described in [docs/SPEC.md](docs/SPEC.md).
