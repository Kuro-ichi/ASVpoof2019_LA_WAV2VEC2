# Wav2Vec2 XLS-R Package Scaffold

This is a production-friendly scaffold to train/evaluate a Wav2Vec2 (e.g. XLS-R/Base) CTC model from Python files.

## Structure
```
wav2vec2_xlsr/
  __init__.py
  config.py
  data/
    __init__.py
    dataset.py
  models/
    __init__.py
    asr.py
  engine/
    __init__.py
    trainer.py
    evaluator.py
  utils/
    __init__.py
    metrics.py
    logger.py
    seed.py
scripts/
  train.py
  eval.py
configs/
  defaults.yaml
tests/
  test_smoke.py
README.md
pyproject.toml
requirements.txt
LICENSE
.gitignore
```

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Train (requires CSV manifests)
python scripts/train.py --config configs/defaults.yaml   --train_csv data/train.csv --val_csv data/val.csv

# Eval
python scripts/eval.py --config configs/defaults.yaml   --val_csv data/val.csv --checkpoint checkpoints/asr.pth
```

### CSV manifest format
The CSV must have headers: `path,text`  
- `path`: path to WAV/FLAC file
- `text`: transcript string

## Notes
- This scaffold uses Hugging Face `transformers` + simple PyTorch `DataLoader`.
- Replace/extend dataset & normalization as needed for your language.
