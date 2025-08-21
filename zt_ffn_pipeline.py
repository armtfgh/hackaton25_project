from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Robust featurizer (self-contained)
# ---------------------------
def _try_import_build_module():
    try:
        import importlib.util
        cand_paths = [
            "build_material_features_2.py",                        # cwd
            str(Path(__file__).parent / "build_material_features_2.py"),  # next to this script
            "build_material_features_2.py",              # common upload path
        ]
        for p in cand_paths:
            if os.path.exists(p):
                spec = importlib.util.spec_from_file_location("build_material_features_2", p)
                bm = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(bm)
                return bm
    except Exception:
        pass
    return None

_BM = _try_import_build_module()

def _formula_to_fraction_vector(formula: str, max_Z: int = 94) -> np.ndarray:
    """
    Convert formula -> 94-length elemental fraction vector.
    Tries your build_material_features_2 implementation first; otherwise uses pymatgen.
    """
    if _BM and hasattr(_BM, "formula_to_fraction_vector"):
        vec = np.array(_BM.formula_to_fraction_vector(formula), dtype=np.float64)
        if vec.shape[0] != max_Z:
            raise ValueError(f"Expected {max_Z}-length vector, got {vec.shape[0]}")
        return vec

    # Fallback: use pymatgen
    try:
        from pymatgen.core.composition import Composition
        from pymatgen.core.periodic_table import Element
    except Exception as e:
        raise ImportError(
            "pymatgen not available and build_material_features_2 not found. "
            "Install pymatgen (`pip install pymatgen`) or place build_material_features_2.py next to this script."
        ) from e

    comp = Composition(formula)
    total_atoms = comp.num_atoms
    vec = np.zeros(max_Z, dtype=np.float64)
    for el, amt in comp.get_el_amt_dict().items():
        Z = Element(el).Z
        if 1 <= Z <= max_Z:
            vec[Z - 1] = amt / total_atoms
    return vec

def _load_properties_matrix(props_csv: str) -> np.ndarray:
    """
    Expect shape (P,94). If user provides (94,P), we auto-transpose.
    Cast to float64; non-numeric coerced to NaN.
    """
    df = pd.read_csv(props_csv)
    # coerce to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    mat = df.to_numpy(dtype=np.float64)
    if mat.shape[1] == 94:
        return mat
    if mat.shape[0] == 94:
        return mat.T
    raise ValueError(f"Expected properties matrix with a 94-dim axis; got {mat.shape}")

def featurize_formula_safe(
    formula: str,
    props_csv: Optional[str] = None,
    use_fractions: bool = True,
    prop_stats: bool = True,
    clip_abs: float = 1e6,
) -> np.ndarray:
    """
    Robust featurization:
    - Fractions 94-d
    - Optional: property statistics per row of props_csv (mean/min/max/mode/std)
    NaN/Inf are handled with np.nansum + masked min/max, and values are clipped.
    """
    frac_vec = _formula_to_fraction_vector(formula)  # (94,)
    blocks = []
    if use_fractions:
        blocks.append(frac_vec)

    if prop_stats and props_csv is not None and os.path.exists(props_csv):
        props = _load_properties_matrix(props_csv)  # (P,94)
        props = np.clip(props, -clip_abs, clip_abs)
        props = np.where(np.isfinite(props), props, np.nan)
        mask = frac_vec > 0

        if mask.any():
            # weighted mean: weights sum to 1
            mean_p = np.nansum(props * frac_vec, axis=1)  # (P,)

            sel = props[:, mask]  # (P, R)
            # nan-aware min/max per row
            def _row_min(arr):
                finite = np.isfinite(arr)
                out = np.zeros(arr.shape[0], dtype=np.float64)
                if arr.size > 0:
                    arr_min = np.where(finite, arr, np.inf)
                    has = finite.any(axis=1)
                    out[has] = np.min(arr_min[has], axis=1)
                return out

            def _row_max(arr):
                finite = np.isfinite(arr)
                out = np.zeros(arr.shape[0], dtype=np.float64)
                if arr.size > 0:
                    arr_max = np.where(finite, arr, -np.inf)
                    has = finite.any(axis=1)
                    out[has] = np.max(arr_max[has], axis=1)
                return out

            min_p = _row_min(sel)
            max_p = _row_max(sel)

            # mode: most abundant element with finite column
            order = np.argsort(-frac_vec)
            mode_p = np.zeros(props.shape[0], dtype=np.float64)
            for idx in order:
                col = props[:, idx]
                if np.isfinite(col).any():
                    mode_p = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
                    break

            # std = sqrt(E[X^2] - (E[X])^2)
            props_sq = np.clip(props * props, 0.0, clip_abs**2)
            mean_sq = np.nansum(props_sq * frac_vec, axis=1)
            var_p = np.maximum(mean_sq - mean_p**2, 0.0)
            std_p = np.sqrt(var_p)
        else:
            P = props.shape[0]
            mean_p = min_p = max_p = mode_p = std_p = np.zeros(P, dtype=np.float64)

        blocks.append(np.stack([mean_p, min_p, max_p, mode_p, std_p], axis=1).reshape(-1))

    if not blocks:
        blocks = [frac_vec]

    return np.concatenate(blocks, axis=0).astype(np.float64)  # (D,)

# ---------------------------
# Data utilities
# ---------------------------
def read_formulas_csv(path: str, formula_col: Optional[str] = None) -> pd.Series:
    """Load a CSV containing formulas. If formula_col is None, use the first column."""
    df = pd.read_csv(path)
    if formula_col is None:
        formula_col = df.columns[0]
    ser = df[formula_col].astype(str).str.strip()
    return ser

def read_targets_csv(
    path: str,
    formula_series: pd.Series,
    temp_col: str = "temperature",
    target_col: str = "zt",
    formula_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: ['formula', 'temperature', 'zt']
    If targets CSV has formula column -> join by formula (text match).
    Else -> align by position/order.
    """
    df_t = pd.read_csv(path)
    if formula_col and formula_col in df_t.columns:
        # normalize formula text
        f_norm = formula_series.str.replace(r"\\s+", "", regex=True)
        df_t["_formula_norm"] = df_t[formula_col].astype(str).str.replace(r"\\s+", "", regex=True)
        merged = pd.DataFrame({"formula": formula_series})
        merged["_formula_norm"] = f_norm
        merged = merged.merge(
            df_t[[ "_formula_norm", temp_col, target_col ]],
            on="_formula_norm",
            how="left",
            validate="one_to_one"
        )
        merged = merged.drop(columns=["_formula_norm"])
    else:
        # positional align
        if len(df_t) < len(formula_series):
            raise ValueError("targets_csv has fewer rows than formulas; cannot align by position.")
        merged = pd.DataFrame({"formula": formula_series})
        merged[temp_col] = df_t[temp_col].values[:len(formula_series)]
        merged[target_col] = df_t[target_col].values[:len(formula_series)]
    # basic checks
    if merged[temp_col].isna().any() or merged[target_col].isna().any():
        missing = merged[ merged[temp_col].isna() | merged[target_col].isna() ]
        raise ValueError(f"Missing temperature/zt for some formulas. Rows:\\n{missing.head()}")
    return merged.rename(columns={temp_col: "temperature", target_col: "zt"})

def build_feature_matrix(
    formulas: pd.Series,
    props_csv: Optional[str] = None,
) -> np.ndarray:
    feats = []
    for f in formulas:
        x = featurize_formula_safe(f, props_csv=props_csv, prop_stats=True, use_fractions=True)
        feats.append(x)
    X = np.vstack(feats)  # (N, D)
    return X

class StandardScalerNP:
    """Simple sklearn-like scaler implemented in NumPy."""
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std_ + self.mean_

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, list]):
        s = cls()
        s.mean_ = np.array(d["mean"], dtype=np.float64)
        s.std_  = np.array(d["std"], dtype=np.float64)
        return s

# ---------------------------
# Torch dataset/model
# ---------------------------
class ZTDataset(Dataset):
    def __init__(self, X: np.ndarray, temp: np.ndarray, y: np.ndarray):
        # concat temperature as the last feature
        self.X = np.hstack([X, temp.reshape(-1, 1)]).astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, d_in: int, width: int = 256, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        d_prev = d_in
        for i in range(depth):
            layers += [
                nn.Linear(d_prev, width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            d_prev = width
        layers += [nn.Linear(d_prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_model(
    X: np.ndarray,
    temp: np.ndarray,
    y: np.ndarray,
    output_dir: str,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[str, str]:
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    early_stop_patience=None

    n_val = max(1, int(N * val_frac))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    X_tr, X_val = X[tr_idx], X[val_idx]
    t_tr, t_val = temp[tr_idx], temp[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    # scale X (including temperature) and y separately
    x_scaler = StandardScalerNP().fit(np.hstack([X_tr, t_tr.reshape(-1,1)]))
    y_scaler = StandardScalerNP().fit(y_tr.reshape(-1,1))

    # --- after (correct) ---
    # 1) scale the concatenated inputs using the same scaler used for fitting
    Xtr_full  = x_scaler.transform(np.hstack([X_tr,  t_tr.reshape(-1, 1)]))  # (Ntr, D+1)
    Xval_full = x_scaler.transform(np.hstack([X_val, t_val.reshape(-1, 1)])) # (Nval, D+1)

    # 2) split back into features and temperature
    Xtr_scaled,  ttr_scaled  = Xtr_full[:, :-1],  Xtr_full[:, -1]
    Xval_scaled, tval_scaled = Xval_full[:, :-1], Xval_full[:, -1]

    # 3) scale targets
    ytr_scaled  = y_scaler.transform(y_tr.reshape(-1, 1)).reshape(-1)
    yval_scaled = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)

    # 4) build datasets (ZTDataset will concatenate X & temp internally)
    ds_tr  = ZTDataset(Xtr_scaled,  ttr_scaled,  ytr_scaled)
    ds_val = ZTDataset(Xval_scaled, tval_scaled, yval_scaled)

    # # datasets
    # ds_tr  = ZTDataset(x_scaler.transform(X_tr), x_scaler.transform(t_tr.reshape(-1,1)).reshape(-1), y_scaler.transform(y_tr.reshape(-1,1)).reshape(-1))
    # ds_val = ZTDataset(x_scaler.transform(X_val), x_scaler.transform(t_val.reshape(-1,1)).reshape(-1), y_scaler.transform(y_val.reshape(-1,1)).reshape(-1))

    

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(d_in=ds_tr.X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = 25
    bad = 0

    for ep in range(1, epochs+1):
        # train
        model.train()
        tr_losses = []
        for xb, yb in dl_tr:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        # val
        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in dl_val:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())
        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = model.state_dict()
            bad = 0
        else:
            bad += 1

        if ep % 10 == 0 or ep == 1:
            print(f"[{ep:04d}] train {tr_loss:.4f} | val {val_loss:.4f}")


        if early_stop_patience is not None and bad >= int(early_stop_patience):
            print(...)
            break

        # if bad >= patience:
        #     print(f"Early stopping at epoch {ep}. Best val MSE: {best_val:.4f}")
        #     break

    if best_state is not None:
        model.load_state_dict(best_state)

    # save artifacts
    os.makedirs(output_dir, exist_ok=True)
    model_path = str(Path(output_dir) / "zt_ffn.pt")
    torch.save(model.state_dict(), model_path)

    with open(Path(output_dir) / "x_scaler.json", "w") as f:
        json.dump(x_scaler.to_dict(), f)
    with open(Path(output_dir) / "y_scaler.json", "w") as f:
        json.dump(y_scaler.to_dict(), f)

    # save meta (input dim etc)
    meta = {"d_in": ds_tr.X.shape[1]}
    with open(Path(output_dir) / "meta.json", "w") as f:
        json.dump(meta, f)

    return model_path, str(Path(output_dir))

def load_model_and_scalers(model_dir: str) -> Tuple[MLP, StandardScalerNP, StandardScalerNP]:
    with open(Path(model_dir) / "meta.json", "r") as f:
        meta = json.load(f)
    d_in = meta["d_in"]

    model = MLP(d_in=d_in)
    state = torch.load(str(Path(model_dir) / "zt_ffn.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with open(Path(model_dir) / "x_scaler.json", "r") as f:
        xs = StandardScalerNP.from_dict(json.load(f))
    with open(Path(model_dir) / "y_scaler.json", "r") as f:
        ys = StandardScalerNP.from_dict(json.load(f))

    return model, xs, ys

# ---------------------------
# High-level predict function
# ---------------------------
def predict_zt_from_formula(
    formula: str,
    temperature: float,
    model_dir: str,
    props_csv: Optional[str] = None,
) -> float:
    """
    Load artifacts from `model_dir`, featurize `formula`, append `temperature`,
    run model, and return zT (original scale).
    """
    # build raw features
    x = featurize_formula_safe(formula, props_csv=props_csv, prop_stats=True, use_fractions=True)  # (D,)
    # load model + scalers
    model, x_scaler, y_scaler = load_model_and_scalers(model_dir)

    # concat temp feature
    X = np.hstack([x, np.array([temperature], dtype=np.float64)]).astype(np.float64).reshape(1, -1)
    Xs = x_scaler.transform(X)  # (1, d_in)

    with torch.no_grad():
        inp = torch.from_numpy(Xs.astype(np.float32))
        pred_scaled = model(inp).numpy().reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_scaled).ravel()[0]
    return float(pred)

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formulas_csv", required=True, help="CSV containing formulas")
    ap.add_argument("--formula_col", default=None, help="Name of the column with formulas (default: first column)")
    ap.add_argument("--props_csv", default=None, help="22x94 elemental properties CSV (optional but recommended)")

    ap.add_argument("--targets_csv", required=True, help="CSV containing temperature and zt (and optionally formula)")
    ap.add_argument("--temp_col", default="temperature", help="Column name for temperature in targets CSV")
    ap.add_argument("--target_col", default="zt", help="Column name for zt in targets CSV")
    ap.add_argument("--targets_formula_col", default=None, help="If provided, join targets on this formula column")

    ap.add_argument("--output_dir", default="./zt_model", help="Where to save the model")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.2)
    args = ap.parse_args()

    formulas = read_formulas_csv(args.formulas_csv, formula_col=args.formula_col)
    tgt = read_targets_csv(
        args.targets_csv,
        formulas,
        temp_col=args.temp_col,
        target_col=args.target_col,
        formula_col=args.targets_formula_col,
    )
    # Featurize
    X = build_feature_matrix(tgt["formula"], props_csv=args.props_csv)
    temp = tgt["temperature"].to_numpy(dtype=np.float64)
    y = tgt["zt"].to_numpy(dtype=np.float64)

    # Train
    model_path, out_dir = train_model(
        X, temp, y,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
    )
    print(f"Saved model to: {model_path}")
    print(f"Artifacts in:   {out_dir}")

if __name__ == "__main__":
    main()
