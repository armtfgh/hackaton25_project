#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, sys, os
from typing import Dict, List
from collections import defaultdict

import numpy as np
import pandas as pd

ELEMENTS_Z1_TO_Z94: List[str] = [
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
    "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn",
    "Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
    "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb",
    "Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th",
    "Pa","U","Np","Pu"
]
SYMBOL_TO_INDEX = {sym: i for i, sym in enumerate(ELEMENTS_Z1_TO_Z94)}  # 0-based

# ---------------- Formula parser ----------------
class FormulaParser:
    def __init__(self, s: str):
        self.s = s.strip()
        self.n = len(self.s)
        self.i = 0

    def _skip_ws_and_dots(self):
        while self.i < self.n and (self.s[self.i].isspace() or self.s[self.i] in "·∙"):
            self.i += 1

    def _parse_number(self) -> float:
        start = self.i
        while self.i < self.n and self.s[self.i].isdigit():
            self.i += 1
        if self.i < self.n and self.s[self.i] == ".":
            self.i += 1
            while self.i < self.n and self.s[self.i].isdigit():
                self.i += 1
        if self.i == start:
            return 1.0
        try:
            return float(self.s[start:self.i])
        except ValueError:
            return 1.0

    def _parse_element(self) -> str:
        if self.i >= self.n or not self.s[self.i].isalpha() or not self.s[self.i].isupper():
            return ""
        start = self.i
        self.i += 1
        if self.i < self.n and self.s[self.i].isalpha() and self.s[self.i].islower():
            self.i += 1
        return self.s[start:self.i]

    def _parse_group(self) -> Dict[str, float]:
        self._skip_ws_and_dots()
        counts: Dict[str, float] = defaultdict(float)
        while self.i < self.n:
            self._skip_ws_and_dots()
            if self.i >= self.n:
                break
            ch = self.s[self.i]
            if ch == "(":
                self.i += 1
                inner = self._parse_group()
                if self.i >= self.n or self.s[self.i] != ")":
                    raise ValueError(f"Unmatched '(' at pos {self.i} in '{self.s}'")
                self.i += 1
                mult = self._parse_number()
                for k, v in inner.items():
                    counts[k] += v * mult
                continue
            if ch == ")":
                break
            el = self._parse_element()
            if not el:
                raise ValueError(f"Unexpected token at pos {self.i}: '{self.s[self.i:self.i+8]}' in '{self.s}'")
            qty = self._parse_number()
            counts[el] += qty
        return counts

    def parse(self) -> Dict[str, float]:
        c = self._parse_group()
        self._skip_ws_and_dots()
        if self.i != self.n and self.s[self.i] != ")":
            rest = self.s[self.i:]
            if rest.strip():
                raise ValueError(f"Could not fully parse near: '{rest}' in '{self.s}'")
        return dict(c)


def formula_to_fraction_vector(formula: str) -> List[float]:
    s = formula.replace("\ufeff", "").strip()
    parser = FormulaParser(s)
    try:
        counts = parser.parse()
    except Exception as e:
        print(f"[WARN] Skip unparsable '{formula}': {e}", file=sys.stderr)
        return [0.0] * 94
    filt: Dict[str, float] = {}
    for el, q in counts.items():
        if el in SYMBOL_TO_INDEX:
            filt[el] = filt.get(el, 0.0) + float(q)
        else:
            print(f"[INFO] Ignoring unsupported element '{el}' in '{formula}'", file=sys.stderr)
    tot = sum(filt.values())
    if tot <= 0:
        return [0.0] * 94
    vec = [0.0] * 94
    for el, q in filt.items():
        vec[SYMBOL_TO_INDEX[el]] = q / tot
    return vec

# ---------------- Loaders ----------------
def read_formulas(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for row in rdr:
            for cell in row:
                s = (cell or "").replace("\ufeff", "").strip()
                if s:
                    out.append(s)
    return out

def load_properties_matrix(props_csv: str) -> np.ndarray:
    df = pd.read_csv(props_csv, header=None)
    if df.shape[1] != 94:
        raise ValueError(f"Expected 94 columns (Z=1..94), got shape {df.shape}")
    return df.to_numpy(dtype=float)  # (22,94)

def load_megnet_embeddings(megnet_json: str) -> np.ndarray:
    with open(megnet_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if len(data) < 95:
            raise ValueError("MEGNet list must have at least 95 entries (index 1..94 used).")
        arr = np.array(data[1:95], dtype=float)  # (94, 16)
    elif isinstance(data, dict):
        emb = data.get("elemental_embedding", data)
        vecs = []
        for sym in ELEMENTS_Z1_TO_Z94:
            if sym not in emb:
                raise ValueError(f"MEGNet JSON missing symbol '{sym}'")
            vecs.append(emb[sym])
        arr = np.array(vecs, dtype=float)  # (94, 16)
    else:
        raise ValueError("Unsupported MEGNet JSON structure.")
    if arr.shape[0] != 94:
        raise ValueError(f"MEGNet embedding rows != 94: {arr.shape}")
    return arr

# ---------------- Speed helpers ----------------
def fractions_matrix_from_formulas(formulas: list) -> np.ndarray:
    cache = {}
    rows = []
    for s in formulas:
        if s in cache:
            rows.append(cache[s])
        else:
            v = np.array(formula_to_fraction_vector(s), dtype=float)
            cache[s] = v
            rows.append(v)
    return np.vstack(rows) if rows else np.zeros((0,94), dtype=float)

def precompute_hist_onehot(values_matrix: np.ndarray):
    K = values_matrix.shape[0]
    onehot = np.zeros((K, 5, 94), dtype=float)
    for k in range(K):
        row = values_matrix[k, :]
        gmin = float(np.nanmin(row))
        gmax = float(np.nanmax(row))
        if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax == gmin:
            onehot[k, 4, :] = 1.0
            continue
        edges = np.linspace(gmin, gmax, num=6)
        idx = np.digitize(row, edges[1:-1], right=False)
        idx = np.clip(idx, 0, 4)
        idx[row == gmax] = 4
        oh = np.eye(5, dtype=float)[idx]  # (94,5)
        onehot[k, :, :] = oh.T
    return onehot  # (K,5,94)

def gather_along_axis2(mat_Px94: np.ndarray, idx_N: np.ndarray) -> np.ndarray:
    P = mat_Px94.shape[0]
    return mat_Px94[np.arange(P)[:, None], idx_N[None, :]]  # (P,N)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formulas", "-f", required=True)
    ap.add_argument("--props", "-p", required=False, help="Required if using --prop-stats or --prop-hist")
    ap.add_argument("--megnet", "-m", required=False, help="Required if using --megnet-stats or --megnet-hist")
    ap.add_argument("--output", "-o", default="material_features.csv")
    ap.add_argument("--include-formula", action="store_true")
    ap.add_argument("--temps", type=str, default="", help="Comma/space-separated temperatures (e.g., '300,400')")
    ap.add_argument("--fast", action="store_true", help="Vectorized multi-formula mode (faster for many formulas)")

    ap.add_argument("--fractions", dest="fractions", action="store_true", help="Include 94 elemental fractions")
    ap.add_argument("--no-fractions", dest="fractions", action="store_false")
    ap.set_defaults(fractions=True)

    ap.add_argument("--prop-stats", dest="prop_stats", action="store_true")
    ap.add_argument("--no-prop-stats", dest="prop_stats", action="store_false")
    ap.set_defaults(prop_stats=False)

    ap.add_argument("--prop-hist", dest="prop_hist", action="store_true")
    ap.add_argument("--no-prop-hist", dest="prop_hist", action="store_false")
    ap.set_defaults(prop_hist=False)

    ap.add_argument("--megnet-stats", dest="megnet_stats", action="store_true")
    ap.add_argument("--no-megnet-stats", dest="megnet_stats", action="store_false")
    ap.set_defaults(megnet_stats=False)

    ap.add_argument("--megnet-hist", dest="megnet_hist", action="store_true")
    ap.add_argument("--no-megnet-hist", dest="megnet_hist", action="store_false")
    ap.set_defaults(megnet_hist=False)

    args = ap.parse_args()

    # Parse temps
    temps_list = []
    if args.temps:
        raw = args.temps.replace(",", " ").split()
        for tok in raw:
            try:
                val = float(tok)
                if abs(val - int(val)) < 1e-12:
                    val = int(val)
                temps_list.append(val)
            except Exception:
                pass

    if not (args.fractions or args.prop_stats or args.prop_hist or args.megnet_stats or args.megnet_hist):
        raise SystemExit("ERROR: Select at least one feature group.")

    need_props = args.prop_stats or args.prop_hist
    need_megnet = args.megnet_stats or args.megnet_hist
    if need_props and not args.props:
        raise SystemExit("ERROR: --props is required for --prop-*")
    if need_megnet and not args.megnet:
        raise SystemExit("ERROR: --megnet is required for --megnet-*")

    formulas = read_formulas(args.formulas)
    N = len(formulas)

    props = load_properties_matrix(args.props) if need_props else None
    megnet = load_megnet_embeddings(args.megnet) if need_megnet else None

    P = props.shape[0] if props is not None else 0
    E = megnet.shape[1] if megnet is not None else 0

    # headers
    headers = []
    if args.fractions: headers += list(ELEMENTS_Z1_TO_Z94)
    if args.prop_stats:
        for p in range(P):
            headers += [f"prop{p+1}_mean", f"prop{p+1}_min", f"prop{p+1}_max", f"prop{p+1}_mode", f"prop{p+1}_std"]
    if args.prop_hist:
        for p in range(P):
            for b in range(1,6):
                headers.append(f"prop{p+1}_hist_bin{b}")
    if args.megnet_stats:
        for j in range(E):
            headers += [f"megnet_dim{j+1}_mean", f"megnet_dim{j+1}_min", f"megnet_dim{j+1}_max", f"megnet_dim{j+1}_mode", f"megnet_dim{j+1}_std"]
    if args.megnet_hist:
        for j in range(E):
            for b in range(1,6):
                headers.append(f"megnet_dim{j+1}_hist_bin{b}")
    if temps_list:
        headers.append("Temp")

    # compute
    rows = []
    if args.fast:
        F = fractions_matrix_from_formulas(formulas)  # (N,94)

        out_blocks = []
        if args.fractions:
            out_blocks.append(F)

        if need_props:
            props_arr = props
            M = props_arr @ F.T                     # (P,N)
            M2 = (props_arr**2) @ F.T               # (P,N)
            V = np.maximum(M2 - M*M, 0.0)
            S = np.sqrt(V)
            mode_idx = np.argmax(F, axis=1)         # (N,)
            MODE = gather_along_axis2(props_arr, mode_idx)  # (P,N)

            mask = F > 0
            props_exp = props_arr[None, :, :]       # (1,P,94)
            mask3 = mask[:, None, :]                # (N,1,94)
            with np.errstate(invalid="ignore"):
                MIN = np.where(mask3, props_exp, np.inf).min(axis=2).T  # (P,N)
                MAX = np.where(mask3, props_exp, -np.inf).max(axis=2).T
            MIN[~np.isfinite(MIN)] = 0.0
            MAX[~np.isfinite(MAX)] = 0.0

            if args.prop_stats:
                PS = np.stack([M, MIN, MAX, MODE, S], axis=2).transpose(1,0,2).reshape(N, -1)
                out_blocks.append(PS)

            if args.prop_hist:
                onehot_p = precompute_hist_onehot(props_arr)            # (P,5,94)
                H = np.einsum("pbr,nr->npb", onehot_p, F)               # (N,P,5)
                out_blocks.append(H.reshape(N, -1))

        if need_megnet:
            MM = F @ megnet                                         # (N,E)
            MM2 = F @ (megnet**2)                                   # (N,E)
            VV = np.maximum(MM2 - MM*MM, 0.0)
            SS = np.sqrt(VV)
            mode_idx = np.argmax(F, axis=1)
            MEGMODE = megnet[mode_idx, :]                           # (N,E)
            mask = F > 0
            mexp = megnet[None, :, :]                               # (1,94,E)
            mask3 = mask[:, :, None]                                # (N,94,1)
            with np.errstate(invalid="ignore"):
                min_e = np.where(mask3, mexp, np.inf).min(axis=1)   # (N,E)
                max_e = np.where(mask3, mexp, -np.inf).max(axis=1)  # (N,E)
            min_e[~np.isfinite(min_e)] = 0.0
            max_e[~np.isfinite(max_e)] = 0.0

            if args.megnet_stats:
                MS = np.hstack([MM, min_e, max_e, MEGMODE, SS])     # (N, 5E)
                out_blocks.append(MS)

            if args.megnet_hist:
                onehot_m = precompute_hist_onehot(megnet.T)         # (E,5,94)
                HM = np.einsum("ebr,nr->neb", onehot_m, F)          # (N,E,5)
                out_blocks.append(HM.reshape(N, -1))

        FEAT = np.hstack(out_blocks) if out_blocks else np.zeros((N,0), dtype=float)

        if temps_list:
            for n, formula in enumerate(formulas):
                base = FEAT[n, :].tolist()
                for T in temps_list:
                    if args.include_formula:
                        rows.append([formula] + base + [T])
                    else:
                        rows.append(base + [T])
        else:
            for n, formula in enumerate(formulas):
                base = FEAT[n, :].tolist()
                if args.include_formula:
                    rows.append([formula] + base)
                else:
                    rows.append(base)

    else:
        # per-formula fallback
        for formula in formulas:
            frac = np.array(formula_to_fraction_vector(formula), dtype=float)
            feats = []
            if args.fractions:
                feats.append(frac)

            if need_props:
                props_arr = props
                mean = props_arr @ frac
                mask = frac > 0
                if np.any(mask):
                    minv = np.nanmin(props_arr[:, mask], axis=1)
                    maxv = np.nanmax(props_arr[:, mask], axis=1)
                else:
                    minv = np.zeros((P,), dtype=float); maxv = np.zeros((P,), dtype=float)
                modev = props_arr[:, int(np.argmax(frac))]
                var = (props_arr**2) @ frac - mean**2
                var = np.maximum(var, 0.0)
                std = np.sqrt(var)
                if args.prop_stats:
                    feats.append(np.stack([mean, minv, maxv, modev, std], axis=1).reshape(-1))
                if args.prop_hist:
                    onehot_p = precompute_hist_onehot(props_arr)
                    histp = np.einsum("pbr,r->pb", onehot_p, frac)
                    feats.append(histp.reshape(-1))

            if need_megnet:
                meanm = frac @ megnet
                mask = frac > 0
                if np.any(mask):
                    sel = megnet[mask, :]
                    minm = np.nanmin(sel, axis=0); maxm = np.nanmax(sel, axis=0)
                else:
                    minm = np.zeros((E,), dtype=float); maxm = np.zeros((E,), dtype=float)
                modem = megnet[int(np.argmax(frac)), :]
                varm = (frac @ (megnet**2)) - meanm**2
                varm = np.maximum(varm, 0.0)
                stdm = np.sqrt(varm)
                if args.megnet_stats:
                    feats.append(np.stack([meanm, minm, maxm, modem, stdm], axis=1).reshape(-1))
                if args.megnet_hist:
                    onehot_m = precompute_hist_onehot(megnet.T)
                    histm = np.einsum("ebr,r->eb", onehot_m, frac)
                    feats.append(histm.reshape(-1))

            feat_vec = np.concatenate(feats, axis=0) if feats else np.array([], dtype=float)
            if temps_list:
                for T in temps_list:
                    if args.include_formula:
                        rows.append([formula] + feat_vec.tolist() + [T])
                    else:
                        rows.append(feat_vec.tolist() + [T])
            else:
                if args.include_formula:
                    rows.append([formula] + feat_vec.tolist())
                else:
                    rows.append(feat_vec.tolist())

    # suffix and write
    parts = []
    if args.fractions: parts.append("F")
    if args.prop_stats: parts.append("PS")
    if args.prop_hist: parts.append("PH")
    if args.megnet_stats: parts.append("MS")
    if args.megnet_hist: parts.append("MH")
    if temps_list: parts.append("T")
    if args.fast: parts.append("FAST")
    suffix = "_".join(parts) if parts else "NONE"
    base, ext = os.path.splitext(args.output)
    out_path = f"{base}__{suffix}{ext or '.csv'}"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if args.include_formula:
            w.writerow(["formula"] + headers)
        else:
            w.writerow(headers)
        w.writerows(rows)

    print(out_path)

if __name__ == "__main__":
    main()
