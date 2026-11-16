"""局部自适应的组合去噪：统计判据 + 半径判据，尺度取自每个点所在平面分块的典型点间距。"""
import numpy as np
from scipy.spatial import cKDTree


def _tile_median(values, xy, tile, min_count=50):
    """每个点所在 tile×tile 平面分块内 values 的中位数；点数不足的分块退回全局中位数。"""
    ij = np.floor((xy - xy.min(0)) / tile).astype(np.int64)
    label = ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]
    order = np.lexsort((values, label))
    _, start, inverse, count = np.unique(label[order], return_index=True, return_inverse=True, return_counts=True)
    median = values[order][start + count // 2]
    median[count < min_count] = np.median(values)
    out = np.empty_like(values)
    out[order] = median[inverse]
    return out


def denoise(xyz, k=16, stat_ratio=6.0, radius_ratio=4.0, min_neighbors=4, tile=10.0, chunk=500_000):
    """返回保留掩码（True = 保留）。满足任一判据即删除：

    统计判据：该点到 k 个近邻的平均距离 > stat_ratio × 所在分块的典型值。
    半径判据：半径 radius_ratio × 典型值 内的近邻数 < min_neighbors。

    “典型值”是该点所在 tile 米平面分块内平均近邻距离的中位数。分块里真实点占多数，
    所以成片的稀疏噪声也会被识别；比值取得宽，林下稀疏的真实地面点不会被删。
    """
    n = len(xyz)
    if n <= k:
        return np.ones(n, bool)
    tree = cKDTree(xyz)
    mean_d, mth_d = np.empty(n, np.float32), np.empty(n, np.float32)
    for s in range(0, n, chunk):  # 分块查询，限制峰值内存
        d, _ = tree.query(xyz[s:s + chunk], k=k + 1, workers=-1)
        mean_d[s:s + chunk] = d[:, 1:].mean(1)
        mth_d[s:s + chunk] = d[:, min_neighbors]
    typical = _tile_median(mean_d, xyz[:, :2], tile)
    return (mean_d <= stat_ratio * typical) & (mth_d <= radius_ratio * typical)
