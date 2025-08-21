
from pathlib import Path
import importlib.util
import numpy as np
from typing import Optional, List

# Import the user's feature builder module from its path
_build_mod_path = "build_material_features_2.py"
_spec = importlib.util.spec_from_file_location("build_material_features_2", _build_mod_path)
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)

def featurize_formula(
    formula: str,
    props_csv: Optional[str] = None,
    megnet_csv: Optional[str] = None,
    use_fractions: bool = True,
    prop_stats: bool = False,
    megnet_stats: bool = False,
) -> List[float]:
    """
    Convert a chemical formula into a feature vector.

    Args:
        formula: e.g., "LiFePO4" or "Ba0.5Sr0.5TiO3".
        props_csv: path to 22x94 CSV where rows are elemental properties and columns Z=1..94.
        megnet_csv: path to 94xE CSV where rows are Z=1..94 and columns are embedding dims.
        use_fractions: include the 94 elemental fractional composition.
        prop_stats: if True and props_csv is given, append mean/min/max/mode/std per property (shape 22*5).
        megnet_stats: if True and megnet_csv is given, append mean/min/max/mode/std per embedding dim (shape E*5).

    Returns:
        A flat Python list[float] of features.
    """
    # Fractions (94,)
    frac_vec = np.array(bm.formula_to_fraction_vector(formula), dtype=float)  # (94,)
    blocks = []
    if use_fractions:
        blocks.append(frac_vec)

    # Elemental property stats
    if prop_stats and props_csv:
        props = bm.load_properties_matrix(props_csv)  # (P=22,94)
        mask = frac_vec > 0
        if mask.any():
            mean_p = props @ frac_vec                    # (22,)
            sel = props[:, mask]                         # (22, R)
            min_p = np.min(sel, axis=1) if sel.size else np.zeros_like(mean_p)
            max_p = np.max(sel, axis=1) if sel.size else np.zeros_like(mean_p)
            mode_idx = int(np.argmax(frac_vec))
            mode_p = props[:, mode_idx]                  # (22,)
            mean_sq = (props**2) @ frac_vec
            var_p = np.maximum(mean_sq - mean_p**2, 0.0)
            std_p = np.sqrt(var_p)
        else:
            P = props.shape[0]
            mean_p = min_p = max_p = mode_p = std_p = np.zeros(P, dtype=float)
        blocks.append(np.stack([mean_p, min_p, max_p, mode_p, std_p], axis=1).reshape(-1))

    # MEGNet embedding stats
    if megnet_stats and megnet_csv:
        megnet = bm.load_megnet_embeddings(megnet_csv)  # (94,E)
        mask = frac_vec > 0
        if mask.any():
            mean_e = (frac_vec @ megnet)                 # (E,)
            sel = megnet[mask, :]                        # (R,E)
            min_e = np.min(sel, axis=0) if sel.size else np.zeros_like(mean_e)
            max_e = np.max(sel, axis=0) if sel.size else np.zeros_like(mean_e)
            mode_idx = int(np.argmax(frac_vec))
            mode_e = megnet[mode_idx, :]                 # (E,)
            mean_sq = (frac_vec @ (megnet**2))           # (E,)
            var_e = np.maximum(mean_sq - mean_e**2, 0.0)
            std_e = np.sqrt(var_e)
        else:
            E = megnet.shape[1]
            mean_e = min_e = max_e = mode_e = std_e = np.zeros(E, dtype=float)
        blocks.append(np.stack([mean_e, min_e, max_e, mode_e, std_e], axis=1).reshape(-1))

    if not blocks:
        blocks = [frac_vec]

    return np.concatenate(blocks, axis=0).astype(float).tolist()
