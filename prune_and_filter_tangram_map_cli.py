#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prune low-confidence cell->spot edges from Tangram ad_map, then drop low-confidence cells
using quantile-based voting; output all intermediate & final results.
python prune_and_filter_tangram_map_cli.py \
  --ad-map /mnt/c/jieguo/GSE111672/PDAC_A_tangram_P6/ad_map_cells_best.h5ad \
  --sp-h5ad /mnt/c/jieguo/GSE111672/PDAC_A/PDAC_A_ad_sp_ready.h5ad \
  --out-dir /mnt/c/jieguo/GSE111672/PDAC_A_tangram_P6/pruned_edges_cells_all \
  --topk-per-row 50 \
  --cumm-mass 0.85 \
  --p-min 0.01 \
  --severity mild \
  --per-type 1 \
  --vote-min 2 \
  --protect-maxp 0.80
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse import save_npz
import matplotlib.pyplot as plt


# ============== 通用工具函数 ==============

def ensure_csr(x):
    """把任意稀疏/稠密矩阵转成 CSR；避免 COO 没有 indptr 的问题。"""
    if sparse.issparse(x):
        return x.tocsr()
    return sparse.csr_matrix(np.asarray(x))


def row_topk_prune_norm(T, topk, cumm, pmin):
    T = ensure_csr(T)
    n, m = T.shape
    indptr, indices, data = T.indptr, T.indices, T.data

    rows, cols, vals = [], [], []

    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        idx_all = indices[s:e]
        val_all = data[s:e]

        if val_all.size == 0:
            continue

        idx = idx_all.copy()
        val = val_all.copy()

        # 1) Top-K
        if val.size > topk:
            sel = np.argpartition(val, -topk)[-topk:]
            idx, val = idx[sel], val[sel]

        # 2) 从大到小排序
        order = np.argsort(-val)
        idx, val = idx[order], val[order]

        # 3) 归一到概率
        p = val / (val.sum() + 1e-12)

        # 4) 累计质量截断
        csum = np.cumsum(p)
        keep = (csum <= cumm) | (np.arange(p.size) == 0)  # 至少保留 top1
        idx, p = idx[keep], p[keep]

        # 5) 去掉太小的边
        mask = p >= pmin
        idx, p = idx[mask], p[mask]

        # 6) 极端情况：全被干掉了，就保留原始行最大值那个 spot
        if p.size == 0:
            j = idx_all[np.argmax(val_all)]
            idx = np.array([j], dtype=int)
            p = np.array([1.0], dtype=float)

        # 7) 再归一一次
        p = p / p.sum()

        rows += [i] * p.size
        cols += idx.tolist()
        vals += p.tolist()

    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, m))


def row_max_csr(T):
    """逐行最大值（考虑稀疏零），返回 shape=(n,) 的 float 数组。"""
    T = ensure_csr(T)
    n = T.shape[0]
    out = np.zeros(n, dtype=float)
    indptr, data = T.indptr, T.data
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        if e > s:
            out[i] = data[s:e].max()
        else:
            out[i] = 0.0
    return out


def entropy_norm_rows(T):
    """归一化熵：对每行概率分布除以 log(非零项数)，返回 (n,)。"""
    T = ensure_csr(T)
    sums = np.asarray(T.sum(axis=1)).ravel() + 1e-12
    Tn = T.multiply(1.0 / sums[:, None]).tocsr()
    out = np.zeros(T.shape[0], dtype=float)

    for i, (s, e) in enumerate(zip(Tn.indptr[:-1], Tn.indptr[1:])):
        p = Tn.data[s:e] + 1e-12
        H = -(p * np.log(p)).sum()
        k = max(e - s, 1)
        out[i] = H / np.log(k + 1e-12)

    return out


def effective_support_norm_rows(T):
    """
    有效支持数（1/∑p^2）再/非零项数，归一到(0,1]。
    这里直接对应你 edge_stats 里 eff_support_norm 的定义。
    """
    T = ensure_csr(T)
    sums = np.asarray(T.sum(axis=1)).ravel() + 1e-12
    Tn = T.multiply(1.0 / sums[:, None]).tocsr()
    out = np.zeros(T.shape[0], dtype=float)

    for i, (s, e) in enumerate(zip(Tn.indptr[:-1], Tn.indptr[1:])):
        p = Tn.data[s:e] + 1e-12
        eff = 1.0 / np.square(p).sum()
        k = max(e - s, 1)
        out[i] = eff / k

    return out


def load_spot_xy(ad_sp):
    """从 ST AnnData 中尽量找到 2D 坐标（用于 soft variance）。"""
    if 'spatial' in ad_sp.obsm_keys():
        M = ad_sp.obsm['spatial']
        if getattr(M, "shape", None) and M.shape[1] >= 2:
            return np.asarray(M)[:, :2]

    if {'x', 'y'}.issubset(ad_sp.obs.columns):
        return ad_sp.obs[['x', 'y']].to_numpy()

    cand = [
        ('array_row', 'array_col'),
        ('pxl_row_in_fullres', 'pxl_col_in_fullres'),
        ('aligned_y', 'aligned_x'),
    ]
    for a, b in cand:
        if {a, b}.issubset(ad_sp.obs.columns):
            return ad_sp.obs[[b, a]].to_numpy()

    return None


def compute_soft_variance(T, spot_xy):
    """
    对每个 cell（行），按照其行概率分布在 spot_xy 上计算加权方差。
    返回 shape=(n_cells,) 的 1D 数组。
    """
    Tn = ensure_csr(T)
    sums = np.asarray(Tn.sum(axis=1)).ravel() + 1e-12
    Tn = Tn.multiply(1.0 / sums[:, None]).tocsr()

    n_cells = Tn.shape[0]
    var = np.zeros(n_cells, dtype=float)

    for i, (s, e) in enumerate(zip(Tn.indptr[:-1], Tn.indptr[1:])):
        cols = Tn.indices[s:e]
        p = Tn.data[s:e]
        if cols.size == 0:
            var[i] = 0.0
            continue
        mu = (p[:, None] * spot_xy[cols]).sum(axis=0)
        dev = spot_xy[cols] - mu
        var[i] = (p[:, None] * (dev ** 2)).sum()

    return var


# ============== 主流程 ==============

def main():
    parser = argparse.ArgumentParser(
        description="Prune Tangram ad_map edges and then drop low-confidence cells using voting."
    )
    parser.add_argument(
        "--ad-map", required=True,
        help="Tangram 输出的 ad_map_cells(.h5ad) 或 ad_map_clusters(.h5ad)，形状为 cells/clusters × spots。"
    )
    parser.add_argument(
        "--sp-h5ad", default=None,
        help="可选：ST h5ad 文件，若提供则尝试从中读取坐标，计算 soft_variance。"
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="输出目录，将在其中写入所有中间和最终结果。"
    )

    # 边级剪枝参数
    parser.add_argument("--topk-per-row", type=int, default=50,
                        help="每行（每个 cell）最多保留的 spot 数量（默认 50）。")
    parser.add_argument("--cumm-mass", type=float, default=0.85,
                        help="在 TopK 内，按照概率累积到该比例为止（默认 0.85）。")
    parser.add_argument("--p-min", type=float, default=0.01,
                        help="行归一后，< p_min 的边会被丢弃（默认 0.01）。")

    # 细胞级过滤参数
    parser.add_argument("--severity", choices=["mild", "strict"], default="mild",
                        help="过滤严格程度（mild: 10–20% 删除; strict: 20–35% 删除，粗略目标）。")
    parser.add_argument("--per-type", type=int, default=1, choices=[0, 1],
                        help="是否按细胞类型分层阈值 (obs['cell_type'])，1=是（默认），0=否。")
    parser.add_argument("--min-degree", type=int, default=2,
                        help="度数兜底阈值（剪边后连边 < 此值 的 cell 会增加一票）。")
    parser.add_argument("--vote-min", type=int, default=None,
                        help="删除所需的最少票数（默认：mild=2, strict=3）。")
    parser.add_argument("--protect-maxp", type=float, default=0.80,
                        help="max_p >= protect_maxp 的细胞会被强制保留。")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_fp = os.path.join(args.out_dir, "run_log.txt")
    open(log_fp, "w").close()

    def log(msg):
        print(msg)
        with open(log_fp, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")

    # ---- 0) 解析 severity 参数 ----
    if args.severity == "mild":
        Q_LOW, Q_HIGH = 0.10, 0.90
        default_vote_min = 2
    else:
        Q_LOW, Q_HIGH = 0.20, 0.80
        default_vote_min = 3

    VOTE_MIN = args.vote_min if args.vote_min is not None else default_vote_min
    PER_TYPE = (args.per_type == 1)
    MIN_DEGREE = args.min_degree
    PROTECT_MAXP = args.protect_maxp

    # ---- 1) 读取 ad_map ----
    log(f"[info] read ad_map: {args.ad_map}")
    ad_map = sc.read_h5ad(args.ad_map)
    T0 = ensure_csr(ad_map.X)
    log(f"[info] T0 shape = {T0.shape}")
    n_cells_before = T0.shape[0]
    n_edges_before = int(T0.nnz)

    # ---- 2) 读取 ST 坐标（可选） ----
    spot_xy = None
    if args.sp_h5ad is not None and os.path.exists(args.sp_h5ad):
        log(f"[info] read ST h5ad for coordinates: {args.sp_h5ad}")
        ad_sp = sc.read_h5ad(args.sp_h5ad)
        spot_xy = load_spot_xy(ad_sp)
        if spot_xy is None:
            log("[warn] failed to infer spot coordinates from ST h5ad; soft_variance will be skipped.")
        else:
            log(f"[info] spot_xy shape = {spot_xy.shape}")
    elif args.sp_h5ad is not None:
        log(f"[warn] sp_h5ad file not found: {args.sp_h5ad}")

    # ---- 3) 边级剪枝 ----
    log("[info] pruning edges (row_topk_prune_norm) ...")
    T1 = row_topk_prune_norm(
        T0,
        topk=args.topk_per_row,
        cumm=args.cumm_mass,
        pmin=args.p_min,
    )
    n_edges_after_prune = int(T1.nnz)
    log(f"[info] edges after prune: {n_edges_after_prune}/{n_edges_before} "
        f"({n_edges_after_prune/n_edges_before:.1%})")

    # 保存剪边后的矩阵 + h5ad（未删 cell）
    save_npz(os.path.join(args.out_dir, "T_cells_by_spots_pruned.npz"), T1)
    ad_map_pruned = ad_map.copy()
    ad_map_pruned.X = T1
    ad_pruned_fp = os.path.join(args.out_dir, "ad_map_cells.pruned.h5ad")
    ad_map_pruned.write(ad_pruned_fp)
    log(f"[save] pruned ad_map (no cell dropped yet): {ad_pruned_fp}")

    # ---- 4) 计算 per-cell 指标（edge_stats 合并在这里）----
    log("[info] computing per-cell metrics from pruned matrix ...")
    max_p = row_max_csr(T1)
    entropy_norm = entropy_norm_rows(T1)
    eff_support_norm = effective_support_norm_rows(T1)
    degree_kept = np.diff(T1.indptr)  # 每行非零数

    cell_ids = np.asarray(ad_map.obs_names.astype(str))
    assert max_p.shape[0] == T1.shape[0] == cell_ids.shape[0]

    stats_df = pd.DataFrame({
        "cell_id": cell_ids,
        "max_p": max_p,
        "entropy_norm": entropy_norm,
        "eff_support_norm": eff_support_norm,
        "degree_kept": degree_kept,
    }).set_index("cell_id", drop=True)

    # soft_variance（如有坐标）
    has_softvar = False
    if spot_xy is not None:
        log("[info] computing soft_variance ...")
        stats_df["soft_variance"] = compute_soft_variance(T1, spot_xy)
        has_softvar = True

    # 存一份“原始 per-cell 指标”
    edge_stats_dir = os.path.join(args.out_dir, "edge_stats")
    os.makedirs(edge_stats_dir, exist_ok=True)
    stats_fp = os.path.join(edge_stats_dir, "per_cell_edge_stats.csv")
    stats_df.to_csv(stats_fp)
    log(f"[save] per-cell edge stats -> {stats_fp}")

    # ---- 5) 按投票规则删细胞 ----

    # 对齐（实际上这里本来就一致，写一遍防御）
    ad_ids = ad_map_pruned.obs_names.astype(str)
    common = ad_ids.intersection(stats_df.index.astype(str))
    if len(common) == 0:
        raise RuntimeError("对齐失败：ad_map 与 per_cell_edge_stats 没有共有的细胞 ID。")
    if len(common) < ad_map_pruned.n_obs or len(common) < stats_df.shape[0]:
        log(f"[warn] 细胞ID不完全一致：共有 {len(common)} / ad {ad_map_pruned.n_obs} / stats {stats_df.shape[0]}，将按交集对齐。")
    ad_filt = ad_map_pruned[common].copy()
    df = stats_df.loc[common].copy()

    # 清理缺失
    need_cols = ["max_p", "entropy_norm", "eff_support_norm", "degree_kept"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"指标缺失：{missing}（内部逻辑错误）")

    n_before = df.shape[0]
    df = df.dropna(subset=need_cols)
    n_after = df.shape[0]
    if n_after < n_before:
        log(f"[warn] 有 {n_before - n_after} 行存在关键指标 NaN，已丢弃。")
        ad_filt = ad_filt[df.index].copy()

    has_softvar = "soft_variance" in df.columns and not df["soft_variance"].isna().all()
    log(f"[info] 对齐后用于投票的细胞数：n_cells={ad_filt.n_obs}；使用列："
        f"{need_cols + (['soft_variance'] if has_softvar else [])}")

    # ---- 5.1) 计算阈值（全局或按 cell_type） ----
    def pick_thresholds(sub: pd.DataFrame):
        thr = {
            "max_p_low": float(sub["max_p"].quantile(Q_LOW)),
            "entropy_hi": float(sub["entropy_norm"].quantile(Q_HIGH)),
            "effnorm_hi": float(sub["eff_support_norm"].quantile(Q_HIGH)),
            "degree_low": int(max(sub["degree_kept"].quantile(Q_LOW), MIN_DEGREE)),
        }
        if has_softvar:
            thr["softvar_hi"] = float(sub["soft_variance"].quantile(Q_HIGH))
        return thr

    thr_by_type = None
    if PER_TYPE and ("cell_type" in ad_filt.obs.columns):
        thr_by_type = {}
        for ct, idx in ad_filt.obs.groupby("cell_type", observed=True).indices.items():
            sub = df.loc[ad_filt.obs_names[idx]]
            if sub.shape[0] == 0:
                continue
            thr_by_type[ct] = pick_thresholds(sub)
        with open(os.path.join(args.out_dir, "thresholds_by_celltype.json"),
                  "w", encoding="utf-8") as f:
            json.dump(thr_by_type, f, ensure_ascii=False, indent=2)
        log(f"[info] 已按细胞类型计算阈值并保存：thresholds_by_celltype.json（类型数={len(thr_by_type)}）")
    else:
        thr_global = pick_thresholds(df)
        with open(os.path.join(args.out_dir, "thresholds_global.json"),
                  "w", encoding="utf-8") as f:
            json.dump(thr_global, f, ensure_ascii=False, indent=2)
        log(f"[info] 已计算全局阈值并保存：thresholds_global.json -> {thr_global}")

    # ---- 5.2) 投票打分 & 强指派保护 ----
    max_p_arr   = df["max_p"].to_numpy()
    entropy_arr = df["entropy_norm"].to_numpy()
    effnorm_arr = df["eff_support_norm"].to_numpy()
    degree_arr  = df["degree_kept"].to_numpy()
    softvar_arr = df["soft_variance"].to_numpy() if has_softvar else None

    votes = np.zeros(df.shape[0], dtype=int)
    protect_mask = (max_p_arr >= PROTECT_MAXP)

    if thr_by_type is not None:
        ct_series = ad_filt.obs.loc[df.index, "cell_type"]
        for ct, sub_idx in ct_series.groupby(ct_series, observed=True).groups.items():
            thr = thr_by_type.get(ct, None)
            if thr is None:
                continue
            ixs = df.index.get_indexer(sub_idx)
            ixs = ixs[ixs >= 0]
            v = np.zeros(ixs.size, dtype=int)
            v += (max_p_arr[ixs]   <= thr["max_p_low"]).astype(int)
            v += (entropy_arr[ixs] >= thr["entropy_hi"]).astype(int)
            v += (effnorm_arr[ixs] >= thr["effnorm_hi"]).astype(int)
            v += (degree_arr[ixs]  <= thr["degree_low"]).astype(int)
            if has_softvar:
                v += (softvar_arr[ixs] >= thr["softvar_hi"]).astype(int)
            votes[ixs] = v
    else:
        thr = thr_global
        votes += (max_p_arr   <= thr["max_p_low"]).astype(int)
        votes += (entropy_arr >= thr["entropy_hi"]).astype(int)
        votes += (effnorm_arr >= thr["effnorm_hi"]).astype(int)
        votes += (degree_arr  <= thr["degree_low"]).astype(int)
        if has_softvar:
            votes += (softvar_arr >= thr["softvar_hi"]).astype(int)

    drop_mask = (votes >= VOTE_MIN) & (~protect_mask)
    keep_mask = ~drop_mask

    keep_ids = df.index[keep_mask].astype(str)
    drop_ids = df.index[drop_mask].astype(str)

    log(f"[result] 细胞保留：{keep_mask.sum()} / {len(keep_mask)} "
        f"({keep_mask.mean():.1%})；删除：{drop_mask.sum()}；保护阈值 max_p >= {PROTECT_MAXP}；VOTE_MIN={VOTE_MIN}")

    # ---- 5.3) 导出过滤后的结果 ----
    cells_filtered_dir = os.path.join(args.out_dir, "cells_filtered")
    os.makedirs(cells_filtered_dir, exist_ok=True)

    ad_cells_filtered = ad_filt[keep_ids].copy()
    out_h5ad_filtered = os.path.join(cells_filtered_dir, "ad_map_cells.pruned.cells_filtered.h5ad")
    ad_cells_filtered.write(out_h5ad_filtered)

    df_out = df.copy()
    df_out["lowconf_votes"] = votes
    df_out["keep"] = keep_mask
    df_out.to_csv(os.path.join(cells_filtered_dir, "per_cell_with_votes.csv"))

    pd.Series(keep_ids, name="cell_id").to_csv(
        os.path.join(cells_filtered_dir, "cells_keep.txt"), index=False)
    pd.Series(drop_ids, name="cell_id").to_csv(
        os.path.join(cells_filtered_dir, "cells_drop.txt"), index=False)

    if "cell_type" in ad_filt.obs.columns:
        grp = pd.DataFrame({
            "keep": keep_mask,
            "drop": drop_mask,
            "ct": ad_filt.obs.loc[df.index, "cell_type"].values
        })
        by = grp.groupby("ct", observed=True).agg(
            n=("keep", "size"),
            keep=("keep", "sum"),
            drop=("drop", "sum"),
        )
        by["keep_rate"] = by["keep"] / by["n"]
        by.to_csv(os.path.join(cells_filtered_dir, "by_celltype_summary_after_filter.csv"))
        log("[save] by_celltype_summary_after_filter.csv")

    with open(os.path.join(cells_filtered_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"SEVERITY={args.severity}, PER_TYPE={PER_TYPE}, "
                f"Q_LOW={Q_LOW}, Q_HIGH={Q_HIGH}, "
                f"MIN_DEGREE={MIN_DEGREE}, VOTE_MIN={VOTE_MIN}, PROTECT_MAXP={PROTECT_MAXP}\n")
        f.write(f"Cells (after edge prune): {n_cells_before}\n")
        f.write(f"Cells after drop        : {keep_mask.sum()}/{len(keep_mask)} "
                f"({keep_mask.mean():.1%}), drop {drop_mask.sum()}\n")
        f.write(f"Edges before prune      : {n_edges_before}\n")
        f.write(f"Edges after prune       : {n_edges_after_prune}\n")

    log("\n[done] 所有结果写入：")
    log("  - 剪边后的 ad_map          : " + ad_pruned_fp)
    log("  - 剪边后的矩阵 (npz)       : T_cells_by_spots_pruned.npz")
    log("  - per-cell edge stats      : edge_stats/per_cell_edge_stats.csv")
    log("  - 过滤后的 adata           : cells_filtered/ad_map_cells.pruned.cells_filtered.h5ad")
    log("  - 每细胞指标+票数          : cells_filtered/per_cell_with_votes.csv")
    log("  - 保留/删除名单            : cells_filtered/cells_keep.txt / cells_filtered/cells_drop.txt")
    log("  - 阈值（全局/分类型）      : thresholds_global.json 或 thresholds_by_celltype.json")
    log("  - 概览汇总                 : cells_filtered/summary.txt")
    if "cell_type" in ad_filt.obs.columns:
        log("  - 分类型保留率             : cells_filtered/by_celltype_summary_after_filter.csv")


if __name__ == "__main__":
    main()
