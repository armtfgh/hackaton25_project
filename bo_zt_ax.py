"""
Bayesian optimization of thermoelectric composition with Ax

This script uses your trained digital twin (predict_zt_from_formula) to optimize
zT at a fixed temperature of 300 K, over the composition parameters of the
prototype (Bi,Sb)(Se,Te,Br)3.

Search space:
  - y_sb in [0, 1]: Sb fraction on the (Bi,Sb) cation sublattice.
  - l_se, l_te, l_br in [-6, 6]: unconstrained "logits" that are mapped to the
    anion sublattice simplex via softmax -> (f_se, f_te, f_br).
    We cap Br at 0.20 and re-normalize Se/Te, then round to 0.02 steps.

Requirements:
  pip install ax-platform botorch gpytorch torch numpy pandas

Usage:
  python bo_zt_ax.py     --model_dir ./zt_model     --props_csv list_of_elemental_properties.csv     --trials 20     --seed 42     --max_halide 0.20     --step 0.02
"""

from __future__ import annotations
import argparse
import numpy as np
from typing import Dict

# Ax Service API
from ax.service.managed_loop import optimize

# Import your predictor from the training pipeline script
# Ensure zt_ffn_pipeline_v2.py (or v1) is in your PYTHONPATH or same folder.
try:
    from zt_ffn_pipeline_v2 import predict_zt_from_formula
except ImportError:
    # fallback to v1 if present
    from zt_ffn_pipeline import predict_zt_from_formula


# -----------------------
# Composition utilities
# -----------------------
def round_frac(x: float, step: float = 0.02) -> float:
    return float(np.clip(np.round(x / step) * step, 0.0, 1.0))

def simplex_from_logits(z):
    z = np.array(z, dtype=float)
    z = z - z.max()  # stabilize
    p = np.exp(z)
    s = p.sum()
    if s <= 0:
        return np.ones_like(p) / len(p)
    return (p / s).astype(float)

def build_formula_with_mixing(y_sb: float, logits_x, step: float = 0.02, max_halide: float = 0.20) -> str:
    """
    (Bi_{1-y} Sb_{y})(Se/Te/Br)_3
    logits_x -> softmax -> (f_se, f_te, f_br)
    Apply halide cap (Br <= max_halide), renormalize Se+Te, quantize to 'step'.
    """
    y = float(np.clip(y_sb, 0.0, 1.0))
    f_se, f_te, f_br = simplex_from_logits(logits_x)
    # Cap Br
    cap = float(np.clip(max_halide, 0.0, 1.0))
    if f_br > cap:
        rest = max(1e-12, 1.0 - f_br)
        scale = (1.0 - cap) / rest
        f_se, f_te, f_br = f_se * scale, f_te * scale, cap
    # Quantize and renormalize
    f_se, f_te, f_br = round_frac(f_se, step), round_frac(f_te, step), round_frac(f_br, step)
    s = f_se + f_te + f_br
    if s <= 0:
        f_se, f_te, f_br = 1.0, 0.0, 0.0  # degenerate fallback
    else:
        f_se, f_te, f_br = f_se / s, f_te / s, f_br / s
    return f"(Bi{1-y:.2f}Sb{y:.2f})(Se{f_se:.2f}Te{f_te:.2f}Br{f_br:.2f})3"


# -----------------------
# Ax optimization driver
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to saved model artifacts (zt_ffn.pt, scalers, meta)")
    ap.add_argument("--props_csv", default=None, help="22x94 elemental properties CSV (optional, but recommended)")
    ap.add_argument("--trials", type=int, default=20, help="Number of BO trials")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--max_halide", type=float, default=0.20, help="Max Br fraction on anion sublattice")
    ap.add_argument("--step", type=float, default=0.02, help="Composition rounding step")
    ap.add_argument("--tempK", type=float, default=300.0, help="Fixed operating temperature (K)")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # Parameters for Ax (y_sb, and 3 logits for (Se, Te, Br))
    parameters = [
        {"name": "y_sb", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "l_se", "type": "range", "bounds": [-6.0, 6.0]},
        {"name": "l_te", "type": "range", "bounds": [-6.0, 6.0]},
        {"name": "l_br", "type": "range", "bounds": [-6.0, 6.0]},
    ]

    def eval_zt_ax(param_dict: Dict[str, float]) -> float:
        y_sb = float(param_dict["y_sb"])
        l_se = float(param_dict["l_se"])
        l_te = float(param_dict["l_te"])
        l_br = float(param_dict["l_br"])
        logits = [l_se, l_te, l_br]
        formula = build_formula_with_mixing(
            y_sb, logits, step=args.step, max_halide=args.max_halide
        )
        zt_hat = predict_zt_from_formula(
            formula=formula,
            temperature=args.tempK,
            model_dir=args.model_dir,
            props_csv=args.props_csv,
        )
        print(f"Trial → {formula} @ {args.tempK:.0f} K  =>  zT̂ = {zt_hat:.4f}")
        return float(zt_hat)

    # Run Ax optimize loop
    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=eval_zt_ax,
        objective_name="zt",
        minimize=False,
        total_trials=args.trials,
        random_seed=args.seed,
    )

    best_zt = float(values[0]["zt"])
    y_sb = float(best_parameters["y_sb"])
    logits = [float(best_parameters["l_se"]), float(best_parameters["l_te"]), float(best_parameters["l_br"])]
    best_formula = build_formula_with_mixing(y_sb, logits, step=args.step, max_halide=args.max_halide)

    print("=== Optimization complete ===")
    print(f"Best predicted zT: {best_zt:.4f} at T={args.tempK:.0f} K")
    print(f"Best parameters: {best_parameters}")
    print(f"Best formula:    {best_formula}")

if __name__ == "__main__":
    main()
