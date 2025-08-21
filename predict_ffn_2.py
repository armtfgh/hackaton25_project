#!/usr/bin/env python3
"""
Inference script for the trained FFN multi-target regressor.

Updates:
- --include-formula : now robust. If a column literally named "Formula" exists, use it.
  Otherwise include the *leftmost* column from the CSV.
- --include-temp / --inclue-temp : include the "Temp" column if present; else include the
  column 6th from the right.
- Suppress torch.load FutureWarning by attempting weights_only=True and falling back cleanly.
- If features drop rows due to NaNs, ground-truth targets (if present) and any included context
  columns are aligned to the kept rows.

Usage:
  python predict_ffn.py --checkpoint ./ffn_run/ffn_checkpoint.pth \
      --csv /path/to/new_or_same_format.csv --out ./predictions.csv \
      --include-formula --include-temp
"""
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
import warnings


class FeedForwardNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_hidden: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(num_hidden):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to ffn_checkpoint.pth from training.")
    p.add_argument("--csv", type=str, required=True, help="Path to CSV for prediction.")
    p.add_argument("--out", type=str, default="./predictions.csv", help="Where to write predictions CSV.")
    # Include options
    p.add_argument("--include-formula", action="store_true", help="Include the leftmost 'Formula' column (or simply the first column) in output.")
    p.add_argument("--include-temp", "--inclue-temp", dest="include_temp", action="store_true",
                   help="Include 'Temp' column if present; otherwise include the column 6th from the right.")
    return p.parse_args()


def zscore_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def safe_load_checkpoint(path: str):
    # Try to avoid FutureWarning by using weights_only=True if supported.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch: no weights_only kw
        return torch.load(path, map_location="cpu")
    except Exception as e:
        warnings.warn(f"Safe load failed ({e}); falling back to default torch.load with pickle.")
        return torch.load(path, map_location="cpu")


def main():
    args = parse_args()
    ckpt = safe_load_checkpoint(args.checkpoint)

    feat_cols: List[str] = ckpt["columns"]["feature_cols"]
    targ_cols: List[str] = ckpt["columns"]["target_cols"]

    x_mean = np.array(ckpt["norm"]["x_mean"], dtype=np.float32)
    x_std = np.array(ckpt["norm"]["x_std"], dtype=np.float32)
    y_mean = np.array(ckpt["norm"]["y_mean"], dtype=np.float32)
    y_std = np.array(ckpt["norm"]["y_std"], dtype=np.float32)

    arch = ckpt["arch"]
    model = FeedForwardNet(
        in_dim=arch["in_dim"],
        out_dim=arch["out_dim"],
        hidden_dim=arch["hidden_dim"],
        num_hidden=arch["num_hidden"],
        dropout=arch["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)
    model.eval()

    df = pd.read_csv(args.csv)

    # Ensure required feature columns are present; coerce to numeric
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in input CSV: {missing}")

    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    mask_valid = ~np.isnan(X).any(axis=1)
    if not mask_valid.all():
        dropped = int((~mask_valid).sum())
        print(f"[warn] Dropping {dropped} rows with NaNs in features for prediction.")
        df = df.loc[mask_valid].reset_index(drop=True)
        X = X[mask_valid]

    Xn = zscore_apply(X, x_mean, x_std).astype(np.float32)

    with torch.no_grad():
        X_tensor = torch.from_numpy(Xn).to(device)
        preds_norm = model(X_tensor).cpu().numpy()
    preds = preds_norm * y_std + y_mean  # invert z-score

    out_cols: List[str] = []
    blocks: List[np.ndarray] = []

    # Optional: include Formula (leftmost column if "Formula" not present)
    if args.include_formula:
        if "Formula" in df.columns:
            formula_col = "Formula"
        else:
            formula_col = df.columns[0]
        out_cols.append(formula_col)
        blocks.append(df[formula_col].astype(str).values.reshape(-1, 1))

    # Optional: include Temp (right-6 or by exact name)
    if args.include_temp:
        temp_col: Optional[str] = None
        if "Temp" in df.columns:
            temp_col = "Temp"
        elif len(df.columns) >= 6:
            temp_col = df.columns[-6]
        if temp_col is None:
            raise ValueError("Requested --include-temp but could not identify the Temp column (need a column named 'Temp' or at least 6 columns).")
        out_cols.append(temp_col)
        blocks.append(pd.to_numeric(df[temp_col], errors="coerce").values.reshape(-1, 1))

    # If ground truth targets exist in the provided CSV, include ACTUAL then PREDICTED for each target
    has_truth = all(c in df.columns for c in targ_cols)
    if has_truth:
        y_true = df[targ_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        # Align ground truth rows if any were dropped by mask_valid
        if y_true.shape[0] != preds.shape[0]:
            y_true = y_true[:preds.shape[0], :]
        for i, c in enumerate(targ_cols):
            out_cols.extend([c, f"{c}_pred"])
            blocks.append(y_true[:, i].reshape(-1, 1))
            blocks.append(preds[:, i].reshape(-1, 1))
    else:
        for i, c in enumerate(targ_cols):
            out_cols.append(f"{c}_pred")
            blocks.append(preds[:, i].reshape(-1, 1))

    out_matrix = np.concatenate(blocks, axis=1) if blocks else preds
    out_df = pd.DataFrame(out_matrix, columns=out_cols)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[saved] Predictions -> {out_path}")


if __name__ == "__main__":
    # Silence only the specific FutureWarning text if it still appears due to older PyTorch
    warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")
    main()
