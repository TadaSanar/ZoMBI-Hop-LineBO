## ZoMBI-Hop-LineBO

DB/serial-driven runner wiring ZoMBI-Hop + LineBO into an external experiment loop.

### Structure

- `src/`: core ZoMBI-Hop package (mirrors `ZoMBI-Hop_final/zombihop/src/`)
- `tests/`: unit tests (mirrors `ZoMBI-Hop_final/zombihop/tests/`)
- `scripts/`: entrypoints + database/serial integration
  - `scripts/main.py`: launches the serial process + optimization process
  - `scripts/run_zombi_main.py`: DB-backed ZoMBI-Hop runner (used by `scripts/main.py`)
  - `scripts/communication.py`: serial + SQLite bridge
  - `scripts/initialize_databases.py`: creates/clears SQLite DBs

### Setup

From `ZoMBI-Hop-LineBO/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

From the repo root:

```bash
python -m scripts.main
python -m scripts.main list
python -m scripts.main <uuid>
```
