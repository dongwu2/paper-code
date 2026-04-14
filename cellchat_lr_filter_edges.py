#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
from pathlib import Path
from typing import Optional, List, Set, Tuple

import numpy as np
import pandas as pd
import anndata as ad


# -----------------------------
# Column candidates
# -----------------------------
CELL_COL_CANDIDATES_SENDER = [
    "cell", "cell_id", "src", "src_cell", "src_cell_id", "src_idx", "src_cell_idx", "sender_cell_id"
]
CELL_COL_CANDIDATES_RECEIVER = [
    "cell", "cell_id", "dst", "dst_cell", "dst_cell_id", "dst_idx", "dst_cell_idx", "receiver_cell_id"
]

LR_COL_CANDIDATES = ["lr", "pair_name", "pair", "lr_name", "LR", "lr_id", "interaction_name"]
LIG_COL_CANDS = ["ligand", "L", "lig"]
REC_COL_CANDS = ["receptor", "R", "rec"]
CT_COL_CANDS  = ["cell_type", "celltype", "cluster", "ctype", "labels", "group"]

SIG_BOOL_CANDS = ["is_sig", "significant", "sig"]
# IMPORTANT: removed single-letter 'q'/'p' to avoid false detection
SIG_QVAL_CANDS = ["qval", "q_value", "fdr", "adj_p", "p_adj", "padj"]
SIG_PVAL_CANDS = ["pval", "p_value"]


# -----------------------------
# Helpers: column detection
# -----------------------------
def detect_col(cols, cands) -> Optional[str]:
    """Case-insensitive; exact match first, then substring fallback (OK for general columns)."""
    lower = {c.lower(): c for c in cols}
    for k in cands:
        if k.lower() in lower:
            return lower[k.lower()]
    for c in cols:
        cl = c.lower()
        if any(k.lower() in cl for k in cands):
            return c
    return None


def detect_col_exact(cols, cands) -> Optional[str]:
    """Case-insensitive EXACT match only (use this for sig columns to avoid mis-detection)."""
    lower = {c.lower(): c for c in cols}
    for k in cands:
        kk = k.lower()
        if kk in lower:
            return lower[kk]
    return None


# -----------------------------
# Helpers: LR normalization
# -----------------------------
def normalize_lr_series(s: pd.Series) -> pd.Series:
    """Canonicalize LR strings to improve overlap."""
    x = s.astype(str).str.upper().str.strip()
    x = x.str.replace(" ", "", regex=False)

    # If format is like "LIG_REC" (single underscore) and no "__", convert to "__"
    mask = (~x.str.contains("__", regex=False)) & (x.str.count("_") == 1)
    x.loc[mask] = x.loc[mask].str.replace("_", "__", regex=False)

    return x


def extract_lr_series(df: pd.DataFrame) -> pd.Series:
    """
    Extract LR as a Series without changing df columns.
    Priority:
      1) lr/pair_name/pair/lr_name/interaction_name...
      2) ligand + receptor
    """
    lr_col = detect_col(df.columns, LR_COL_CANDIDATES)
    if lr_col is not None:
        return normalize_lr_series(df[lr_col])

    lig = detect_col(df.columns, LIG_COL_CANDS)
    rec = detect_col(df.columns, REC_COL_CANDS)
    if lig is None or rec is None:
        raise KeyError("Cannot find LR column nor ligand/receptor columns in dataframe.")
    lr = df[lig].astype(str) + "__" + df[rec].astype(str)
    return normalize_lr_series(lr)


# -----------------------------
# Significance handling (fixed)
# -----------------------------
def significance_mask(sig: pd.DataFrame, alpha: float) -> pd.Series:
    """
    Decide which rows are significant.
    - boolean is_sig preferred (EXACT match)
    - else q-value <= alpha (EXACT match)
    - else p-value <= alpha (EXACT match)
    - else: assume already-thresholded => all True
    """
    is_sig_col = detect_col_exact(sig.columns, SIG_BOOL_CANDS)
    if is_sig_col is not None:
        return sig[is_sig_col].astype(bool)

    qcol = detect_col_exact(sig.columns, SIG_QVAL_CANDS)
    if qcol is not None:
        q = pd.to_numeric(sig[qcol], errors="coerce")
        return q.fillna(np.inf) <= alpha

    pcol = detect_col_exact(sig.columns, SIG_PVAL_CANDS)
    if pcol is not None:
        p = pd.to_numeric(sig[pcol], errors="coerce")
        return p.fillna(np.inf) <= alpha

    # No indicator columns => treat all rows as significant
    return pd.Series(True, index=sig.index)


def canonicalize_sig_pairs(sig: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Return 2-col significant pairs ['lr','cell_type']."""
    sig = sig.copy()
    sig["lr"] = extract_lr_series(sig)

    ct = detect_col(sig.columns, CT_COL_CANDS)
    if ct is None:
        raise KeyError("Significance table missing cell_type column (e.g., cell_type/cluster/labels).")
    sig = sig.rename(columns={ct: "cell_type"})

    m = significance_mask(sig, alpha)
    sig = sig.loc[m]
    return sig[["lr", "cell_type"]].drop_duplicates()


def load_sig_lr_set(path: str, alpha: float) -> Set[str]:
    """Load sig file and return the set of significant LR (row-wise, with the fixed mask)."""
    sig = pd.read_csv(path)
    lr = extract_lr_series(sig)
    m = significance_mask(sig, alpha)
    lr = lr.loc[m]
    return set(lr.dropna().unique().tolist())


# -----------------------------
# Default mode: attach cell types
# -----------------------------
def attach_celltypes(edges: pd.DataFrame,
                     cellmap: pd.Series,
                     side: str,
                     override_col: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """
    Attach 'cell_type' by mapping per-row cell id to obs[celltype].
    Returns: (df_with_lr_and_cell_type, original_cell_col_name)
    """
    n_obs = len(cellmap)

    # choose cell column
    if override_col is not None and override_col in edges.columns:
        ccol = override_col
        print(f"[{side}] using explicit cell column: {ccol}")
    else:
        ccol = detect_col(edges.columns,
                          CELL_COL_CANDIDATES_SENDER if side == "sender" else CELL_COL_CANDIDATES_RECEIVER)
        if ccol is None:
            raise KeyError(f"{side} edges: cannot find a cell id/index column "
                           f"(pass --{side}-cell-col to specify explicitly)")
        print(f"[{side}] detected cell column: {ccol}")

    df = edges.copy()
    df["lr"] = extract_lr_series(df)

    # map cell_id -> cell_type
    ser = df[ccol]
    obs_index = cellmap.index

    if ser.dtype == object and ser.isin(obs_index).mean() > 0.9:
        print(f"[{side}] mapping by obs_names (string IDs)")
        df["cell_type"] = ser.map(cellmap)
    else:
        idx = pd.to_numeric(ser, errors="coerce")
        if idx.notna().all():
            idx = idx.astype(int)
            if (0 <= idx).all() and (idx < n_obs).all():
                print(f"[{side}] mapping by positional indices (0..n_obs-1)")
                arr = cellmap.astype(str).values
                df["cell_type"] = pd.Series(arr[idx.values], index=df.index)
            else:
                raise ValueError(f"{side}: numeric cell ids out of bounds "
                                 f"(min={idx.min()}, max={idx.max()}, n_obs={n_obs}).")
        else:
            print(f"[{side}] attempting relaxed string mapping to obs_names")
            df["cell_type"] = ser.astype(str).map(cellmap)

    missing = df["cell_type"].isna().sum()
    if missing:
        print(f"[warn] {side}: {missing} edges have cell ids not found; they will be dropped.")
        df = df.dropna(subset=["cell_type"])

    return df, ccol


def filter_edges_by_pairs(df: pd.DataFrame, sig_pairs: pd.DataFrame) -> pd.DataFrame:
    """Row-filter per-cell edges by (lr, cell_type)."""
    key = sig_pairs.set_index(["lr", "cell_type"]).index
    idx = pd.MultiIndex.from_frame(df[["lr", "cell_type"]])
    return df.loc[idx.isin(key)]


# -----------------------------
# LR-only mode
# -----------------------------
def filter_edges_by_lr_only(edges_in: str, edges_out: str, lr_keep: Set[str]) -> None:
    df = pd.read_csv(edges_in)
    lr = extract_lr_series(df)
    keep = df.loc[lr.isin(lr_keep)].copy()
    Path(edges_out).parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(edges_out, index=False)
    print(f"[save] {edges_out} | {len(keep)}/{len(df)} kept")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Filter RAW edges using CellChat/CPDB-like gating.")
    ap.add_argument("--lr-only", action="store_true",
                    help="PURE LR-level filtering (ignore cell types).")

    # cell meta (default mode only)
    ap.add_argument("--cell-h5ad", default=None, help="Required unless --lr-only")
    ap.add_argument("--celltype-key", default="cell_type")

    # edges
    ap.add_argument("--sender-edges-in", required=True)
    ap.add_argument("--receiver-edges-in", required=True)
    ap.add_argument("--sender-edges-out", required=True)
    ap.add_argument("--receiver-edges-out", required=True)

    # sig tables
    ap.add_argument("--sender-sig", required=True)
    ap.add_argument("--receiver-sig", required=True)
    ap.add_argument("--alpha", type=float, default=0.10, help="Cutoff if using p/q columns")

    # LR-level both-sides
    ap.add_argument("--require-both-sides", action="store_true",
                    help="Keep LR significant on both sides (intersection). Otherwise union in lr-only mode.")

    # default mode only: ultra-broad cull
    ap.add_argument("--broad-frac", type=float, default=None,
                    help="Cull LR with significant cell_types fraction >= frac (default mode).")
    ap.add_argument("--broad-max", type=int, default=None,
                    help="Cull LR with significant cell_types count > max (default mode).")

    # default mode only: explicit cell columns
    ap.add_argument("--sender-cell-col", default=None)
    ap.add_argument("--receiver-cell-col", default=None)

    args = ap.parse_args()

    # -------------------------
    # LR-only mode
    # -------------------------
    if args.lr_only:
        s_lr = load_sig_lr_set(args.sender_sig, args.alpha)
        r_lr = load_sig_lr_set(args.receiver_sig, args.alpha)

        if args.require_both_sides:
            lr_keep = s_lr & r_lr
            print(f"[lr-only both-sides] lr_keep={len(lr_keep)} (intersection)")
        else:
            lr_keep = s_lr | r_lr
            print(f"[lr-only union] lr_keep={len(lr_keep)} (union)")

        filter_edges_by_lr_only(args.sender_edges_in, args.sender_edges_out, lr_keep)
        filter_edges_by_lr_only(args.receiver_edges_in, args.receiver_edges_out, lr_keep)
        return

    # -------------------------
    # Default mode (lr, cell_type) gating
    # -------------------------
    if args.cell_h5ad is None:
        raise ValueError("Default mode requires --cell-h5ad (or use --lr-only).")

    A = ad.read_h5ad(args.cell_h5ad)
    if args.celltype_key not in A.obs.columns:
        raise KeyError(f"obs['{args.celltype_key}'] not found; available: {list(A.obs.columns)}")
    cellmap = A.obs[args.celltype_key].astype(str)
    n_ctypes = int(cellmap.nunique())
    print(f"[cells] n={A.n_obs}, cell_types={n_ctypes}")

    s_sig = canonicalize_sig_pairs(pd.read_csv(args.sender_sig), args.alpha)
    r_sig = canonicalize_sig_pairs(pd.read_csv(args.receiver_sig), args.alpha)
    print(f"[sig] sender pairs kept: {len(s_sig)} | receiver pairs kept: {len(r_sig)}")

    # BOTH-SIDES gate at LR level
    if args.require_both_sides:
        lr_both = set(s_sig["lr"].unique()) & set(r_sig["lr"].unique())
        s_sig = s_sig[s_sig["lr"].isin(lr_both)]
        r_sig = r_sig[r_sig["lr"].isin(lr_both)]
        print(f"[both-sides] LR kept: {len(lr_both)}")

    # ULTRA-BROAD cull (optional)
    def cull_broad(sig_df: pd.DataFrame, side: str) -> pd.DataFrame:
        if args.broad_max is None and args.broad_frac is None:
            return sig_df
        counts = sig_df.groupby("lr")["cell_type"].nunique()
        keep = set(counts.index)
        if args.broad_max is not None:
            keep = {lr for lr in keep if counts[lr] <= args.broad_max}
        if args.broad_frac is not None:
            thr = int(np.floor(args.broad_frac * n_ctypes))
            keep = {lr for lr in keep if counts[lr] <= thr}
        out = sig_df[sig_df["lr"].isin(keep)]
        removed = len(set(sig_df["lr"].unique()) - set(out["lr"].unique()))
        print(f"[broad-{side}] removed LR: {removed}")
        return out

    s_sig = cull_broad(s_sig, "sender")
    r_sig = cull_broad(r_sig, "receiver")

    # load & annotate RAW edges
    s_edges_raw = pd.read_csv(args.sender_edges_in)
    r_edges_raw = pd.read_csv(args.receiver_edges_in)

    s_edges, _ = attach_celltypes(s_edges_raw, cellmap, side="sender", override_col=args.sender_cell_col)
    r_edges, _ = attach_celltypes(r_edges_raw, cellmap, side="receiver", override_col=args.receiver_cell_col)

    # filter by (lr, cell_type)
    s_keep = filter_edges_by_pairs(s_edges, s_sig).copy()
    r_keep = filter_edges_by_pairs(r_edges, r_sig).copy()

    # write outputs preserving original column order (no extra lr/cell_type columns)
    s_cols_out = [c for c in s_edges_raw.columns if c in s_keep.columns]
    r_cols_out = [c for c in r_edges_raw.columns if c in r_keep.columns]
    s_keep = s_keep[s_cols_out]
    r_keep = r_keep[r_cols_out]

    Path(args.sender_edges_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.receiver_edges_out).parent.mkdir(parents=True, exist_ok=True)

    s_keep.to_csv(args.sender_edges_out, index=False)
    r_keep.to_csv(args.receiver_edges_out, index=False)

    print(f"[save] sender edges: {len(s_keep)} / {len(s_edges_raw)} -> {args.sender_edges_out}")
    print(f"[save] receiver edges: {len(r_keep)} / {len(r_edges_raw)} -> {args.receiver_edges_out}")


if __name__ == "__main__":
    main()
