#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# ------------------ I/O ------------------

def read_csv_any(path: Path) -> pd.DataFrame:
    for enc in (None, "utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(path, encoding="latin-1", engine="python", low_memory=False)

def detect_cols(df: pd.DataFrame):
    low = {c.lower(): c for c in df.columns}
    cands_s = ["src_spot_idx","src","spot_src","i","from","source"]
    cands_t = ["dst_spot_idx","dst","spot_dst","j","to","target"]
    s = next((low[c] for c in cands_s if c in low), None)
    t = next((low[c] for c in cands_t if c in low), None)
    if s is None or t is None:
        raise KeyError(f"无法在 {list(df.columns)} 中找到 spot 索引列（如 src_spot_idx/dst_spot_idx）")
    return s, t

# ------------------ 核心计算 ------------------

def build_adj(n_nodes: int, src: np.ndarray, dst: np.ndarray, undirected: bool=True) -> csr_matrix:
    data = np.ones_like(src, dtype=np.float32)
    A = csr_matrix((data, (src, dst)), shape=(n_nodes, n_nodes))
    if undirected:
        A = A.maximum(A.T)
    # 去自环
    A.setdiag(0); A.eliminate_zeros()
    return A

def all_pairs_spd(A: csr_matrix, max_spd: int) -> np.ndarray:
    """
    返回 NxN 的最短路距离矩阵（int16），超过 max_spd 或不可达记为 -1。
    """
    # SciPy 会返回 float，inf 表示不可达；unweighted=True 用 BFS
    D = shortest_path(A, directed=False, unweighted=True, return_predecessors=False)
    # 转整型并截断
    D = np.where(np.isfinite(D), D, np.inf)
    D = D.astype(np.float32)
    # 超过 max_spd 的统一标成 -1，便于筛选
    D = np.where((D >= 1) & (D <= max_spd), D, -1)
    return D.astype(np.int16)

def iter_pairs_at_k(D: np.ndarray, k: int):
    """
    生成距离等于 k 的 (i,j) 无序对（i<j）。
    """
    idx = np.where(D == k)
    I, J = idx[0], idx[1]
    mask = I < J
    return I[mask], J[mask]

def neighbors_list(A: csr_matrix):
    """将稀疏邻接转为 list[set]，用于公共邻居计数。"""
    A = A.tocsr()
    rows = []
    for i in range(A.shape[0]):
        start, end = A.indptr[i], A.indptr[i+1]
        rows.append(set(A.indices[start:end]))
    return rows

def count_common_neighbors(nbrs, i, j):
    return len(nbrs[i].intersection(nbrs[j]))

def make_df_edges(i_arr, j_arr, add_cols: dict | None = None) -> pd.DataFrame:
    df = pd.DataFrame({"src_spot_idx": i_arr.astype(np.int64),
                       "dst_spot_idx": j_arr.astype(np.int64)})
    if add_cols:
        for k, v in add_cols.items():
            df[k] = v
    return df

# ------------------ 主流程 ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neighbor-csv", required=True, help="spot↔spot 邻接边 CSV（如 edges_spot_neighbor_spot.csv）")
    ap.add_argument("--n-spots", type=int, default=None, help="可选：显式给出 spot 数；默认按边的最大索引+1 推断")
    ap.add_argument("--max-spd", type=int, default=3, help="构造 spdist1..K（默认 3）")
    ap.add_argument("--undirected", action="store_true", help="将邻接视为无向并对称化（推荐）")
    ap.add_argument("--emit-all", action="store_true", help="额外输出合并版 ALL CSV（含 rel/spd/weight）")
    ap.add_argument("--weight-mode", choices=["none","inv","exp"], default="none",
                    help="可选为每条边附权重：inv=1/spd；exp=exp(-spd/tau)")
    ap.add_argument("--tau", type=float, default=1.0, help="weight=exp(-spd/tau) 的尺度")
    ap.add_argument("--add-common-neighbors", action="store_true",
                    help="为输出边计算公共邻居数 cn（可能略慢）")
    ap.add_argument("--out-dir", required=True, help="输出目录")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dfN = read_csv_any(Path(args.neighbor_csv))
    cS, cT = detect_cols(dfN)

    src = pd.to_numeric(dfN[cS], errors="coerce").dropna().astype(np.int64).to_numpy()
    dst = pd.to_numeric(dfN[cT], errors="coerce").dropna().astype(np.int64).to_numpy()
    n_spots = args.n_spots if args.n_spots is not None else int(max(src.max(), dst.max())) + 1

    print(f"[neighbor] edges={len(src)}  spots={n_spots}  undirected={bool(args.undirected)}")
    A = build_adj(n_spots, src, dst, undirected=args.undirected)
    print(f"[adj] nnz={A.nnz}  shape={A.shape}")

    # 全对最短路（对 N=3.6k 尺度安全）
    D = all_pairs_spd(A, max_spd=args.max_spd)
    print(f"[spd] computed all-pairs shortest path (int16); max_spd={args.max_spd}")

    # 需要的话预先准备公共邻居
    nbrs = neighbors_list(A) if args.add_common_neighbors else None

    # 为 ALL 合并做累积
    frames = []

    for k in range(1, args.max_spd + 1):
        I, J = iter_pairs_at_k(D, k)
        print(f"[spdist{k}] pairs={len(I):,}")
        # 权重
        if args.weight_mode == "inv":
            w = np.full_like(I, 1.0 / float(k), dtype=np.float32)
        elif args.weight_mode == "exp":
            w = np.full_like(I, float(np.exp(-k / max(args.tau, 1e-6))), dtype=np.float32)
        else:
            w = None

        # 公共邻居（可作为 edge_attr）
        cn = None
        if nbrs is not None:
            cn = np.fromiter((count_common_neighbors(nbrs, int(i), int(j)) for i, j in zip(I, J)),
                             dtype=np.int32, count=len(I))

        add_cols = {}
        if w is not None:
            add_cols["weight"] = w
        add_cols["spd"] = np.full_like(I, k, dtype=np.int16)
        if cn is not None:
            add_cols["cn"] = cn

        dfk = make_df_edges(I, J, add_cols)
        # 单独文件：edges_spot_spdist{k}.csv
        out_k = out_dir / f"edges_spot_spdist{k}.csv"
        dfk.to_csv(out_k, index=False)
        print(f"[save] {out_k} | rows={len(dfk):,}")

        if args.emit_all:
            dfk_all = dfk.copy()
            dfk_all["rel"] = f"spdist{k}"
            frames.append(dfk_all)

    if args.emit_all:
        ALL = pd.concat(frames, axis=0, ignore_index=True)
        out_all = out_dir / "edges_spot_spstruct.ALL.csv"
        ALL.to_csv(out_all, index=False)
        print(f"[save] {out_all} | rows={len(ALL):,}")

if __name__ == "__main__":
    main()
