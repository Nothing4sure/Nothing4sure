from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.metrics import classification_report
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel


def fetch_prices(symbol: str, start: str, end: str) -> pd.Series:
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"No close-price data found for {symbol}.")
    return df["Close"].dropna()


def create_sequences(close_prices: pd.Series, window_size: int, num_bins: int) -> np.ndarray:
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna().to_numpy()
    if len(log_returns) <= window_size + 1:
        raise ValueError("Not enough data points for the selected window size.")

    clipped = np.clip(log_returns, np.percentile(log_returns, 1), np.percentile(log_returns, 99))
    bin_edges = np.linspace(clipped.min(), clipped.max(), num_bins + 1)
    tokens = np.digitize(clipped, bin_edges[1:-1], right=False)

    sequences = []
    for i in range(window_size, len(tokens)):
        sequences.append(tokens[i - window_size : i + 1])
    return np.asarray(sequences, dtype=np.int64)


def direction_from_token(token_id: int, num_bins: int) -> str:
    mid = num_bins // 2
    if token_id < mid - 1:
        return "down"
    if token_id > mid + 1:
        return "up"
    return "flat"


def train_model(
    sequences: np.ndarray,
    num_bins: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_dir: Path,
) -> dict:
    split = int(len(sequences) * 0.8)
    train_seq = torch.tensor(sequences[:split], dtype=torch.long)
    test_seq = torch.tensor(sequences[split:], dtype=torch.long)

    config = GPT2Config(
        vocab_size=num_bins,
        n_positions=sequences.shape[1],
        n_layer=2,
        n_head=2,
        n_embd=64,
    )
    model = GPT2LMHeadModel(config)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        perm = torch.randperm(train_seq.size(0))
        train_seq = train_seq[perm]
        epoch_loss = 0.0

        for i in range(0, train_seq.size(0), batch_size):
            batch = train_seq[i : i + batch_size]
            out = model(input_ids=batch, labels=batch)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss / train_seq.size(0):.6f}")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for seq in test_seq:
            context = seq[:-1].unsqueeze(0)
            logits = model(input_ids=context).logits[0, -1]
            pred_token = int(torch.argmax(logits).item())
            true_token = int(seq[-1].item())
            y_true.append(direction_from_token(true_token, num_bins))
            y_pred.append(direction_from_token(pred_token, num_bins))

    report = classification_report(
        y_true,
        y_pred,
        labels=["down", "flat", "up"],
        output_dict=True,
        zero_division=0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "model")
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small GPT-style model for Indian stock direction.")
    parser.add_argument("--symbol", default="^NSEI", help="Yahoo Finance symbol (default: NIFTY 50 index).")
    parser.add_argument("--start", default="2012-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=str(date.today()), help="End date (YYYY-MM-DD).")
    parser.add_argument("--window-size", type=int, default=30, help="Number of past steps used as context.")
    parser.add_argument("--num-bins", type=int, default=64, help="Number of discrete return bins.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--output-dir", default="outputs", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    close_prices = fetch_prices(args.symbol, args.start, args.end)
    sequences = create_sequences(close_prices, args.window_size, args.num_bins)
    report = train_model(
        sequences=sequences,
        num_bins=args.num_bins,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=Path(args.output_dir),
    )
    print("Validation classification report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
