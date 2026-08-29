# Indian Stock Market LLM Starter

This folder contains a minimal starter project to train a small GPT-style model on Indian stock index data.

## What it does

- Downloads price history from Yahoo Finance (default: NIFTY 50 `^NSEI`)
- Converts daily log-returns to discrete tokens
- Trains a lightweight GPT-style causal language model
- Predicts next-step market direction (`down` / `flat` / `up`)
- Saves model and evaluation metrics

## Setup

```bash
cd /home/runner/work/Nothing0g/Nothing0g/stock_llm_india
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --symbol ^NSEI --start 2012-01-01 --epochs 3 --output-dir outputs
```

For another Indian market symbol, pass a different Yahoo ticker (example: `RELIANCE.NS`, `TCS.NS`, `^BSESN`).

## Outputs

- `outputs/model/` → trained model weights/config
- `outputs/metrics.json` → classification metrics on validation split

## Notes

- This is an educational baseline, not financial advice.
- Market prediction is noisy and non-stationary; use stronger validation and risk controls before any real use.
