# Python virtual environment

This project uses a local `.venv/` (gitignored) with **Python 3.11**.
`kaggle-environments>=1.28.0` requires Python 3.11+, so the macOS system
`python3` (3.9) is not sufficient — use Homebrew's `python3.11`.

## First-time setup

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Activate / deactivate

```bash
source .venv/bin/activate   # activate (prompt prefix: (.venv))
deactivate                   # leave the venv
```

## Verify

```bash
python3 -c "import torch; print('torch OK:', torch.__version__); print('MPS:', torch.backends.mps.is_available())"
python3 -c "from kaggle_environments import make; e = make('orbit_wars'); print('kaggle OK')"
```

On Apple Silicon, `MPS: True` confirms the Metal backend is available — pass
`--device mps` to scripts that accept a device flag.

## Smoke test

```bash
python3 -m agents.transformer_v1.runner --smoke-test --device cpu \
    --ckpt data/runs/action/<run-id>/action_best.pt
```

Run-output checkpoints under `data/runs/` are gitignored — train locally or
sync from GCS before running the smoke test.

## Updating dependencies

After editing `requirements.txt`:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```
