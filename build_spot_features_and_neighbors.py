#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
import warnings


# ===== 小工具 =====
def get_spot_xy(ad_sp):
    """优先从 obsm['spatial'] 取坐标，否则尝试常见 obs 列。"""
    if "spatial" in ad_sp.obsm_keys():
        XY = np.asarray(ad_sp.obsm["spatial"])
        if XY.shape[1] >= 2:
            return XY[:, :2]
    for a, b in [("array_col", "array_row"),
                 ("x", "y"),
                 ("aligned_x", "aligned_y")]:
        if a in ad_sp.obs.columns and b in ad_sp.obs.columns:
            return ad_sp.obs[[a, b]].to_numpy()
    return None


def build_radius_graph(XY, k_min=6):
    """
    用自适应半径构建无向 Spot–Spot 图：
    - 对每个点取第 k_min 个最近邻
    - 距离中位数 * 1.05 作为 radius
    """
    nn = NearestNeighbors(n_neighbors=k_min + 1, algorithm="kd_tree").fit(XY)
    dists, _ = nn.kneighbors(XY)
    base = float(np.median(dists[:, k_min]))  # 第 k_min 个最近邻（排除自身）
    radius = base * 1.05
    W = radius_neighbors_graph(XY, radius=radius, mode="connectivity", include_self=False)
    W = W.maximum(W.T)
    return W.tocsr(), radius


def neighbor_mean(W, X):
    """W: (n×n) 0/1 邻接；X: (n×d) -> 邻居均值 (n×d)"""
    W = W.tocsr()
    deg = np.asarray(W.sum(axis=1)).ravel()
    S = W.dot(X)
    out = np.zeros_like(S, dtype=float)
    mask = deg > 0
    out[mask] = S[mask] / deg[mask, None]
    out[~mask] = 0.0
    return out


def zscore_cols(A):
    A = np.asarray(A, dtype=float)
    m = np.nanmean(A, axis=0, keepdims=True)
    s = np.nanstd(A, axis=0, keepdims=True) + 1e-8
    return (A - m) / s


def laplacian_positional_encoding(W, k=16):
    """对称归一化图拉普拉斯的前 k 个非平凡特征向量作为位置编码。"""
    W = W.tocsr()
    n = W.shape[0]
    deg = np.asarray(W.sum(axis=1)).ravel()

    D_inv_sqrt = np.zeros_like(deg, dtype=float)
    with np.errstate(divide='ignore'):
        D_inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
    D_inv_sqrt = sparse.diags(D_inv_sqrt)

    L = sparse.eye(n, format="csr") - D_inv_sqrt @ W @ D_inv_sqrt
    kk = min(k + 1, n - 1)
    if kk <= 0:
        return np.zeros((n, 0), dtype=float)

    _, vecs = eigsh(L, k=kk, which="SM")
    if vecs.shape[1] > 1:
        vecs = vecs[:, 1:1 + k]
    else:
        vecs = np.zeros((n, k), dtype=float)

    return zscore_cols(vecs)


def get_block_from_act(ad, key):
    """从 spot_pathway_tf_acts.h5ad 中取一个 obsm block + 列名。"""
    if key not in ad.obsm_keys():
        return None, []
    X = np.asarray(ad.obsm[key])
    cols = []
    if "feature_blocks" in ad.uns and key in ad.uns["feature_blocks"]:
        cols = list(map(str, ad.uns["feature_blocks"][key].get("cols", [])))
    if not cols or len(cols) != X.shape[1]:
        cols = [f"{key}_{i:02d}" for i in range(X.shape[1])]
    return X, cols


def save_block(ad, key, X, cols, source):
    """保存一块特征到 obsm + 记录 meta 到 uns['feature_blocks']。"""
    if X is None:
        return
    ad.obsm[key] = np.asarray(X, dtype=np.float32)
    ad.uns.setdefault("feature_blocks", {})
    ad.uns["feature_blocks"][key] = {
        "shape": [ad.n_obs, X.shape[1]],
        "source": source,
        "cols": list(map(str, cols))
    }


def export_neighbor_edges(W, XY, out_csv):
    """导出 spot↔spot 邻接边表。"""
    from pathlib import Path
    W = W.tocoo()
    rows = W.row
    cols = W.col
    vals = W.data.astype(float)

    mask = rows != cols
    rows, cols, vals = rows[mask], cols[mask], vals[mask]
    if len(rows) == 0:
        print("[neighbor] W 里没有任何非自环边，跳过导出。")
        return

    lo = np.minimum(rows, cols)
    hi = np.maximum(rows, cols)
    dists = np.linalg.norm(XY[lo] - XY[hi], axis=1)

    df = pd.DataFrame({
        "src_spot_idx": lo.astype(np.int64),
        "dst_spot_idx": hi.astype(np.int64),
        "connectivity": vals,
        "distance": dists.astype(float)
    })

    df = (df.sort_values(["src_spot_idx", "dst_spot_idx", "connectivity"],
                         ascending=[True, True, False])
            .groupby(["src_spot_idx", "dst_spot_idx"], as_index=False)
            .agg(connectivity=("connectivity", "max"),
                 distance=("distance", "min")))

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[neighbor] save {out_path} | edges={len(df):,}")


# ===== 新增：表达 PCA =====
def make_expr_adata(ad_sp, expr_source="auto", expr_layer=None):
    """
    返回一个用于表达 PCA 的 AnnData（obs 同步，var 为表达基因）。
    expr_source: auto/raw/X/layer
    """
    if expr_source == "auto":
        expr_source = "raw" if ad_sp.raw is not None else "X"

    if expr_source == "raw":
        if ad_sp.raw is None:
            raise ValueError("expr_source=raw 但 adata.raw 为 None。")
        X = ad_sp.raw.X
        var = ad_sp.raw.var.copy()
        out = sc.AnnData(X=X, obs=ad_sp.obs.copy(), var=var)
        out.var_names = out.var_names.astype(str)
        return out, "raw"

    if expr_source == "layer":
        if expr_layer is None:
            raise ValueError("expr_source=layer 需要 --expr-layer 指定层名。")
        if expr_layer not in ad_sp.layers.keys():
            raise KeyError(f"在 adata.layers 里找不到 layer='{expr_layer}'，现有: {list(ad_sp.layers.keys())}")
        X = ad_sp.layers[expr_layer]
        out = sc.AnnData(X=X, obs=ad_sp.obs.copy(), var=ad_sp.var.copy())
        out.var_names = out.var_names.astype(str)
        return out, f"layer:{expr_layer}"

    # default: X
    out = sc.AnnData(X=ad_sp.X, obs=ad_sp.obs.copy(), var=ad_sp.var.copy())
    out.var_names = out.var_names.astype(str)
    return out, "X"


def build_expr_pca(
    ad_sp,
    n_pcs=50,
    n_hvg=2000,
    target_sum=1e4,
    expr_source="auto",
    expr_layer=None,
    hvg_flavor="auto",
    do_normalize=True,
    do_log1p=True,
    do_scale=True,
    scale_max_value=10.0,
):
    """
    生成 X_sp_expr_pca: (n_spots, n_pcs)
    - 如果用 raw/layer(通常是 counts)，默认用 seurat_v3 选 HVG，再 normalize+log1p，再 PCA
    - 如果用 X(通常是 lognorm)，默认用 seurat 选 HVG
    """
    ad_expr, src_tag = make_expr_adata(ad_sp, expr_source=expr_source, expr_layer=expr_layer)

    # 选择 HVG flavor
    if hvg_flavor == "auto":
        # raw/layer 更像 counts，适合 seurat_v3；X 多为 lognorm，适合 seurat
        hvg_flavor = "seurat_v3" if src_tag.startswith("raw") or src_tag.startswith("layer") else "seurat"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # HVG：seurat_v3 通常基于 counts（在 normalize/log1p 前跑更合理）
        if n_hvg and n_hvg > 0:
            if hvg_flavor == "seurat_v3":
                sc.pp.highly_variable_genes(ad_expr, n_top_genes=int(n_hvg), flavor="seurat_v3")
                hvg_mask = ad_expr.var.get("highly_variable", None)
                if hvg_mask is None:
                    raise RuntimeError("HVG(seurat_v3) 失败：未生成 var['highly_variable']")
                # 下面再做 normalize/log1p/pca
            else:
                # seurat 需要在 lognorm 上做 HVG
                if do_normalize:
                    sc.pp.normalize_total(ad_expr, target_sum=float(target_sum))
                if do_log1p:
                    sc.pp.log1p(ad_expr)
                sc.pp.highly_variable_genes(ad_expr, n_top_genes=int(n_hvg), flavor=hvg_flavor)

        # 如果走 seurat_v3：HVG 已经选好，但还没 normalize/log1p
        if hvg_flavor == "seurat_v3":
            if do_normalize:
                sc.pp.normalize_total(ad_expr, target_sum=float(target_sum))
            if do_log1p:
                sc.pp.log1p(ad_expr)

        # scale + PCA
        if do_scale:
            sc.pp.scale(ad_expr, max_value=float(scale_max_value))
        sc.tl.pca(ad_expr, n_comps=int(n_pcs), use_highly_variable=(n_hvg and n_hvg > 0))

    X_pca = np.asarray(ad_expr.obsm["X_pca"], dtype=float)
    X_pca = zscore_cols(X_pca)

    cols = [f"expr_pca_{i:02d}" for i in range(1, X_pca.shape[1] + 1)]
    source = f"expr={src_tag}; hvg_flavor={hvg_flavor}; hvg={n_hvg}; normalize={do_normalize}; log1p={do_log1p}; pca={n_pcs}"
    return X_pca.astype(np.float32), cols, source


# ===== CLI =====
def parse_args():
    ap = argparse.ArgumentParser(
        description="Build spot_features.h5ad & spot_features.final.h5ad (with Expression PCA block)."
    )
    ap.add_argument("--sp-h5ad", type=str, required=True,
                    help="ST h5ad，包含 spot 坐标（建议用带 raw 的 with_spatial.h5ad）")
    ap.add_argument("--spot-acts-h5ad", type=str, required=True,
                    help="spot_pathway_tf_acts.h5ad（包含 X_sp_pathway_progeny14 / X_sp_tfact_dorothea）")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="输出目录，将写 spot_features.h5ad / spot_features.final.h5ad / columns.csv")

    # 图构建 & PE 参数
    ap.add_argument("--k-min-neighbors", type=int, default=6)
    ap.add_argument("--pe-dim", type=int, default=16)
    ap.add_argument("--no-coord2d", action="store_true")

    # 导出 spot↔spot 邻接边
    ap.add_argument("--neighbor-out-csv", type=str, default=None)

    # ===== 新增：表达 PCA 参数 =====
    ap.add_argument("--expr-source", type=str, default="auto",
                    choices=["auto", "raw", "X", "layer"],
                    help="表达矩阵来源：auto(优先 raw)/raw/X/layer")
    ap.add_argument("--expr-layer", type=str, default=None,
                    help="当 expr-source=layer 时指定 layers 名称")
    ap.add_argument("--expr-n-pcs", type=int, default=50,
                    help="表达 PCA 维度（默认 50）")
    ap.add_argument("--expr-hvg", type=int, default=2000,
                    help="表达 PCA 使用的 HVG 数（<=0 表示不用 HVG，直接全基因 PCA）")
    ap.add_argument("--expr-target-sum", type=float, default=1e4,
                    help="normalize_total target_sum（默认 1e4）")
    ap.add_argument("--expr-hvg-flavor", type=str, default="auto",
                    choices=["auto", "seurat_v3", "seurat"],
                    help="HVG flavor：auto(raw/layer->seurat_v3, X->seurat)")
    ap.add_argument("--expr-no-normalize", action="store_true",
                    help="关闭 normalize_total（默认开启）")
    ap.add_argument("--expr-no-log1p", action="store_true",
                    help="关闭 log1p（默认开启）")
    ap.add_argument("--expr-no-scale", action="store_true",
                    help="关闭 scale（默认开启）")
    ap.add_argument("--expr-scale-max", type=float, default=10.0,
                    help="scale max_value（默认 10）")

    return ap.parse_args()


def main():
    args = parse_args()

    SP_AD    = args.sp_h5ad
    SP_ACT   = args.spot_acts_h5ad
    OUTDIR   = args.out_dir
    CORE_H5  = os.path.join(OUTDIR, "spot_features.h5ad")
    FINAL_H5 = os.path.join(OUTDIR, "spot_features.final.h5ad")
    CSV_COLS = os.path.join(OUTDIR, "spot_features.final.columns.csv")
    os.makedirs(OUTDIR, exist_ok=True)

    K_MIN_NEIGHBORS = args.k_min_neighbors
    PE_K            = args.pe_dim
    INCLUDE_COORD2D = not args.no_coord2d

    # ===== 读取与对齐 =====
    print("[info] read ST:", SP_AD)
    ad_sp = sc.read_h5ad(SP_AD)
    print("[info] read spot acts:", SP_ACT)
    ad_act = sc.read_h5ad(SP_ACT)

    # 以活性文件为锚点（ad_act.obs_names 就是 spot 顺序）
    if list(ad_sp.obs_names) != list(ad_act.obs_names):
        ad_sp = ad_sp[ad_act.obs_names].copy()
    print(f"[align] spots={ad_sp.n_obs}")

    # ===== 取 spot 级 PROGENy / TF =====
    Xs_pw, s_pw_cols = get_block_from_act(ad_act, "X_sp_pathway_progeny14")
    Xs_tf, s_tf_cols = get_block_from_act(ad_act, "X_sp_tfact_dorothea")
    if Xs_pw is None and Xs_tf is None:
        raise RuntimeError("在 spot_pathway_tf_acts.h5ad 中找不到 spot 级 PROGENy/TF 特征。")

    # ===== 新增：表达 PCA block =====
    print("[expr] building expression PCA for spots ...")
    X_expr, expr_cols, expr_source = build_expr_pca(
        ad_sp,
        n_pcs=args.expr_n_pcs,
        n_hvg=args.expr_hvg,
        target_sum=args.expr_target_sum,
        expr_source=args.expr_source,
        expr_layer=args.expr_layer,
        hvg_flavor=args.expr_hvg_flavor,
        do_normalize=(not args.expr_no_normalize),
        do_log1p=(not args.expr_no_log1p),
        do_scale=(not args.expr_no_scale),
        scale_max_value=args.expr_scale_max,
    )
    print("[ok] X_sp_expr_pca:", X_expr.shape, "|", expr_source)

    # ===== 构建 Spot–Spot 图，Niche + PE =====
    XY = get_spot_xy(ad_sp)
    if XY is None:
        print("[warn] 未找到坐标，跳过 Niche 与位置编码，仅保留活性+表达PCA。")
        W = None
    else:
        W, radius = build_radius_graph(XY, k_min=K_MIN_NEIGHBORS)
        print(f"[info] spot graph built: >= {K_MIN_NEIGHBORS} nn, radius≈{radius:.3f}, edges={W.nnz}")
        if args.neighbor_out_csv:
            export_neighbor_edges(W, XY, args.neighbor_out_csv)

    Xn_pw = zscore_cols(neighbor_mean(W, Xs_pw)) if (W is not None and Xs_pw is not None) else None
    Xn_tf = zscore_cols(neighbor_mean(W, Xs_tf)) if (W is not None and Xs_tf is not None) else None
    n_pw_cols = [f"niche_r1:{c}" for c in s_pw_cols] if Xn_pw is not None else []
    n_tf_cols = [f"niche_r1:{c}" for c in s_tf_cols] if Xn_tf is not None else []

    X_pe, pe_cols = None, []
    if W is not None and PE_K > 0:
        print(f"[info] computing Laplacian positional encodings (k={PE_K}) ...")
        X_pe = laplacian_positional_encoding(W, k=PE_K)
        pe_cols = [f"pe_{i+1:02d}" for i in range(X_pe.shape[1])]

    X_xy, xy_cols = None, []
    if INCLUDE_COORD2D and XY is not None:
        X_xy = zscore_cols(XY[:, :2])
        xy_cols = ["coord_x", "coord_y"]

    # ===== 保存 spot_features.h5ad（分块）=====
    ad_out = sc.AnnData(obs=pd.DataFrame(index=ad_sp.obs_names))
    ad_out.obsm = {}
    ad_out.uns = {"feature_blocks": {}}

    save_block(ad_out, "X_sp_expr_pca", X_expr, expr_cols, expr_source)
    save_block(ad_out, "X_sp_pathway_progeny14", Xs_pw, s_pw_cols,
               "imported: spot_pathway_tf_acts.h5ad (PROGENy)")
    save_block(ad_out, "X_sp_tfact_dorothea", Xs_tf, s_tf_cols,
               "imported: spot_pathway_tf_acts.h5ad (DoRothEA TF)")
    save_block(ad_out, "X_sp_niche_progeny_r1", Xn_pw, n_pw_cols,
               f"neighbor mean on spot graph (radius≈auto, >={K_MIN_NEIGHBORS} nn)")
    save_block(ad_out, "X_sp_niche_tfact_r1", Xn_tf, n_tf_cols,
               f"neighbor mean on spot graph (radius≈auto, >={K_MIN_NEIGHBORS} nn)")
    save_block(ad_out, "X_sp_pe", X_pe, pe_cols,
               f"Laplacian positional encodings (k={PE_K})")
    save_block(ad_out, "X_sp_coord2d", X_xy, xy_cols,
               "z-scored (x,y) coordinates")

    ad_out.write(CORE_H5)
    print("[save]", CORE_H5)

    # ===== 拼接到 X，写出 final =====
    order = [
        ("X_sp_expr_pca",          "expr_pca"),
        ("X_sp_pathway_progeny14", "progeny"),
        ("X_sp_tfact_dorothea",    "tfact"),
        ("X_sp_niche_progeny_r1",  "niche_progeny"),
        ("X_sp_niche_tfact_r1",    "niche_tfact"),
        ("X_sp_pe",                "pe"),
        ("X_sp_coord2d",           "coord"),
    ]

    mats, names, blocks = [], [], []
    for key, pref in order:
        if key not in ad_out.obsm_keys():
            continue
        X = np.asarray(ad_out.obsm[key])
        fb = ad_out.uns["feature_blocks"].get(key, {})
        cols = fb.get("cols", [f"{pref}_{i:02d}" for i in range(X.shape[1])])
        mats.append(X.astype(np.float32))
        names.extend(list(map(str, cols)))
        blocks.extend([key] * X.shape[1])
        print(f"[concat] {key}: {X.shape}")

    if not mats:
        raise RuntimeError("没有可拼接的特征块。")

    X_all = np.concatenate(mats, axis=1).astype(np.float32)

    ad_final = sc.AnnData(
        X=X_all,
        obs=ad_out.obs.copy(),
        var=pd.DataFrame({"block": blocks}, index=pd.Index(names, name="feature"))
    )
    ad_final.obsm = ad_out.obsm.copy()
    ad_final.uns  = ad_out.uns.copy()

    offsets, s = {}, 0
    for key, _ in order:
        d = sum(1 for b in blocks if b == key)
        if d > 0:
            offsets[key] = {"start": int(s), "end": int(s + d), "dim": int(d)}
            s += d
    ad_final.uns["X_blocks"] = offsets
    ad_final.uns["X_summary"] = {
        "n_spots": int(ad_final.n_obs),
        "n_features": int(ad_final.n_vars),
        "blocks": list(offsets.keys()),
        "note": "X = [ExprPCA || PROGENy || TF || Niche(PROGENy/TF) || PE || (x,y)] ; 无 QC、无 cell-type 占比。"
    }

    pd.DataFrame({"feature": ad_final.var_names, "block": ad_final.var["block"]}).to_csv(CSV_COLS, index=False)
    print("[save]", CSV_COLS)

    ad_final.write(FINAL_H5)
    print("[save]", FINAL_H5)
    print("[done] Spot 节点特征（含表达PCA）+（可选）邻接边 已完成。")


if __name__ == "__main__":
    main()
