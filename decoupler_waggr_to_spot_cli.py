#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decoupler 2.1.1 版：用 WAGGR 计算 PROGENy / DoRothEA 活性，并按 Tangram 映射聚合到 spot

- 在单细胞层面算 pathway / TF 活性（WAGGR）
- 利用 Tangram 的 cell×spot 概率矩阵，把活性聚合到 spot
"""

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


# ========= 小工具 =========

def ensure_csr(x):
    return x.tocsr() if sparse.issparse(x) else sparse.csr_matrix(np.asarray(x))


def zscore_block(M):
    M = np.asarray(M, dtype=float)
    mu = np.nanmean(M, axis=0, keepdims=True)
    sd = np.nanstd(M, axis=0, keepdims=True) + 1e-8
    return (M - mu) / sd


# ========= CLI 解析 =========

def parse_args():
    ap = argparse.ArgumentParser(
        description="Run decoupler WAGGR (PROGENy + DoRothEA) on scRNA and aggregate to spots via Tangram mapping."
    )
    ap.add_argument("--sc-h5ad", type=str, required=True,
                    help="scRNA h5ad 文件路径（cells × genes）")
    ap.add_argument("--sp-h5ad", type=str, required=True,
                    help="ST h5ad 文件路径（spots × genes，仅用于对齐 spot 顺序）")
    ap.add_argument("--map-h5ad", type=str, required=True,
                    help="Tangram ad_map h5ad（cell×spot 概率矩阵，行和≈1）")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="输出目录")

    ap.add_argument("--organism", type=str, default="human",
                    help="物种：human 或 mouse（用于 PROGENy / DoRothEA，默认 human）")
    ap.add_argument("--progeny-top", type=int, default=500,
                    help="PROGENy: 每个通路保留的 top 靶基因数（默认 500）")
    ap.add_argument("--doro-levels", type=str, default="A,B,C",
                    help="DoRothEA 置信等级，逗号分隔（默认 A,B,C）")
    ap.add_argument("--min-targets", type=int, default=5,
                    help="WAGGR: 每个 pathway/TF 至少匹配到的基因数 tmin（默认 5）")
    ap.add_argument("--tf-topk", type=int, default=20,
                    help="保留方差最大的前 K 个 TF（默认 20，<=0 表示不过滤）")
    ap.add_argument("--waggr-times", type=int, default=1,
                    help="WAGGR 的 times 参数，=1 时不做置换检验（默认 1）")
    ap.add_argument("--seed", type=int, default=123,
                    help="随机种子（默认 123）")

    return ap.parse_args()


# ========= 主流程 =========

def main():
    args = parse_args()

    SC_AD    = args.sc_h5ad
    SP_AD    = args.sp_h5ad
    MAP_AD   = args.map_h5ad
    OUTDIR   = args.out_dir
    ORGANISM = args.organism
    PROGENY_TOP = args.progeny_top
    DORO_LEVELS = [lv.strip() for lv in args.doro_levels.split(",") if lv.strip()]
    MIN_TARGETS = args.min_targets
    TF_TOPK     = args.tf_topk
    WAGGR_TIMES = args.waggr_times
    SEED        = args.seed

    os.makedirs(OUTDIR, exist_ok=True)

    print("[info] read scRNA:", SC_AD)
    ad_sc = sc.read_h5ad(SC_AD)
    print("[info] read ST   :", SP_AD)
    ad_sp = sc.read_h5ad(SP_AD)
    print("[info] read map  :", MAP_AD)
    ad_map = sc.read_h5ad(MAP_AD)  # Tangram：cell×spot 概率矩阵（行和≈1）

    # —— 对齐为映射顺序（以 map 为主） ——
    cells_map = ad_map.obs_names.astype(str)
    spots_map = ad_map.var_names.astype(str)

    # 细胞对齐：只保留既在 map 又在 sc 中的细胞，并按 map 顺序排序
    sc_obs_names_str = ad_sc.obs_names.astype(str)
    keep_cells = [i for i in cells_map if i in sc_obs_names_str]
    ad_sc = ad_sc[keep_cells].copy()
    # 重新裁剪 map 的 obs 顺序与 ad_sc 对齐
    ad_map = ad_map[ad_sc.obs_names]

    # spot 对齐：如 ST 中有同名 spot，则按 map 顺序裁剪
    sp_obs_names_str = ad_sp.obs_names.astype(str)
    if len(set(spots_map) & set(sp_obs_names_str)) > 0:
        keep_spots = [j for j in spots_map if j in sp_obs_names_str]
        ad_sp = ad_sp[keep_spots].copy()
        ad_map = ad_map[:, ad_sp.obs_names]

    print(f"[info] aligned: n_cells={ad_sc.n_obs}, n_spots={ad_map.n_vars}")

    # —— 基因名统一为大写，方便与网络 target 匹配 ——
    ad_sc_upper = ad_sc.copy()
    ad_sc_upper.var_names = ad_sc_upper.var_names.astype(str).str.upper()

    # ========= decoupler 资源与方法 =========
    import decoupler as dc
    print("[info] decoupler version:", getattr(dc, "__version__", "unknown"))

    # 取 PROGENy / DoRothEA（新 API 在 op.* 命名空间）
    net_pw = dc.op.progeny(organism=ORGANISM, top=PROGENY_TOP)      # source/pathway, target, weight
    net_tf = dc.op.dorothea(organism=ORGANISM, levels=DORO_LEVELS)  # source/TF, target, weight(or mor)

    # 统一 target 大写，DoRothEA 的 mor -> weight
    net_pw["target"] = net_pw["target"].astype(str).str.upper()
    if "mor" in net_tf.columns and "weight" not in net_tf.columns:
        net_tf = net_tf.rename(columns={"mor": "weight"})
    net_tf["target"] = net_tf["target"].astype(str).str.upper()

    # ========= 关键优化 1：只保留 PROGENy + DoRothEA 的 target 基因 =========
    targets_union = set(net_pw["target"]) | set(net_tf["target"])
    genes_sc = set(ad_sc_upper.var_names.astype(str))
    genes_keep = sorted(genes_sc & targets_union)

    print(f"[filter] subset sc genes to union of PROGENy + DoRothEA targets: {len(genes_sc)} -> {len(genes_keep)}")
    if len(genes_keep) < MIN_TARGETS:
        raise RuntimeError(f"可用 target 基因太少（{len(genes_keep)}），检查物种 / 基因名 / doro-levels 是否正确。")

    ad_sc_upper = ad_sc_upper[:, genes_keep].copy()

    # ========= 关键优化 2：表达矩阵转为 float32，减半内存 =========
    from scipy import sparse as _sp
    import numpy as _np

    X = ad_sc_upper.X
    if _sp.issparse(X):
        ad_sc_upper.X = X.astype(_np.float32)
    else:
        ad_sc_upper.X = _np.asarray(X, dtype=_np.float32)

    print("[info] after filtering: sc shape =", ad_sc_upper.shape, ", dtype =", ad_sc_upper.X.dtype)

    # ========= WAGGR =========
    # times=1: 快速版（无置换）；times>1: 置换版，内存消耗 ~ times 倍
    label = f"times={WAGGR_TIMES}" if WAGGR_TIMES > 1 else "times=1 (no permutations)"
    print(f"[run] PROGENy via dc.mt.waggr ({label})...")
    dc.mt.waggr(
        data=ad_sc_upper,
        net=net_pw,
        tmin=MIN_TARGETS,
        times=WAGGR_TIMES,
        seed=SEED,
        verbose=False,
    )
    # 拿回 scores（obs×sources）
    PW_ad = dc.pp.get_obsm(ad_sc_upper, key="score_waggr")
    PW = pd.DataFrame(PW_ad.X, index=PW_ad.obs_names, columns=PW_ad.var_names)
    ad_sc_upper.obsm.pop("score_waggr", None)

    print(f"[run] DoRothEA via dc.mt.waggr ({label})...")
    dc.mt.waggr(
        data=ad_sc_upper,
        net=net_tf,
        tmin=MIN_TARGETS,
        times=WAGGR_TIMES,
        seed=SEED,
        verbose=False,
    )
    TF_ad = dc.pp.get_obsm(ad_sc_upper, key="score_waggr")
    TF = pd.DataFrame(TF_ad.X, index=TF_ad.obs_names, columns=TF_ad.var_names)
    ad_sc_upper.obsm.pop("score_waggr", None)

    # ========= Z-score & 可选裁剪 TF 维度 =========
    X_pw = zscore_block(PW.values)

    if TF_TOPK and TF_TOPK > 0 and TF.shape[1] > TF_TOPK:
        var_tf = TF.var(axis=0)
        keep_tf = var_tf.sort_values(ascending=False).head(TF_TOPK).index
        TF = TF[keep_tf]
        print(f"[filter] select top-{TF_TOPK} TF by variance: {TF.shape[1]} TFs kept.")
    X_tf = zscore_block(TF.values)

    # ========= 保存 cell-level 活性 =========
    ad_cell = sc.AnnData(obs=pd.DataFrame(index=ad_sc.obs_names))
    ad_cell.obsm["X_pathway_progeny14"] = X_pw
    ad_cell.obsm["X_tfact_dorothea"] = X_tf
    ad_cell.uns = {
        "feature_blocks": {
            "X_pathway_progeny14": {
                "shape": [ad_cell.n_obs, X_pw.shape[1]],
                "source": f"decoupler.mt.waggr(times={WAGGR_TIMES}) + PROGENy({ORGANISM}, top={PROGENY_TOP})",
                "cols": [f"progeny_{c}" for c in PW.columns.astype(str)],
            },
            "X_tfact_dorothea": {
                "shape": [ad_cell.n_obs, X_tf.shape[1]],
                "source": f"decoupler.mt.waggr(times={WAGGR_TIMES}) + DoRothEA(levels={DORO_LEVELS})",
                "cols": [f"tfact_{c}" for c in TF.columns.astype(str)],
            },
        }
    }
    out_cell = os.path.join(OUTDIR, "cell_pathway_tf_acts.h5ad")
    ad_cell.write(out_cell)
    print("[save]", out_cell)

    # ========= 聚合到 spot（spot = P^T @ cell_scores）=========
    # Tangram：ad_map.X 为 cell×spot 概率矩阵，可将任意 cell 注释投到空间。
    print("[agg] aggregate cell scores to spots via P^T ...")
    P = ensure_csr(ad_map.X)                 # (cells × spots)
    PW_mat = PW.loc[ad_cell.obs_names].values
    TF_mat = TF.loc[ad_cell.obs_names].values

    SP_pw = P.T.dot(PW_mat)   # (spots × pathways)
    SP_tf = P.T.dot(TF_mat)   # (spots × TFs)

    SP_pw = zscore_block(SP_pw)
    SP_tf = zscore_block(SP_tf)

    ad_sp_out = sc.AnnData(obs=pd.DataFrame(index=ad_map.var_names))
    ad_sp_out.obsm = {
        "X_sp_pathway_progeny14": SP_pw,
        "X_sp_tfact_dorothea": SP_tf,
    }
    ad_sp_out.uns = {
        "feature_blocks": {
            "X_sp_pathway_progeny14": {
                "shape": [ad_sp_out.n_obs, SP_pw.shape[1]],
                "source": f"cell-level waggr(times={WAGGR_TIMES}) aggregated by P^T",
                "cols": [f"progeny_{c}" for c in PW.columns.astype(str)],
            },
            "X_sp_tfact_dorothea": {
                "shape": [ad_sp_out.n_obs, SP_tf.shape[1]],
                "source": f"cell-level waggr(times={WAGGR_TIMES}) aggregated by P^T",
                "cols": [f"tfact_{c}" for c in TF.columns.astype(str)],
            },
        }
    }
    out_sp = os.path.join(OUTDIR, "spot_pathway_tf_acts.h5ad")
    ad_sp_out.write(out_sp)
    print("[save]", out_sp)

    print(f"[done] pathway/TF 活性（cell & spot）已完成 (waggr times={WAGGR_TIMES}).")


if __name__ == "__main__":
    main()
