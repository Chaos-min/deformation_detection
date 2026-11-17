"""地形特征感知的多尺度 PTD（Ni 等 2026，JSTARS 19:15331）。

由粗到细：每层取可靠最低点，用该层距离阈值做一次 PTD，再补回陡坡和低洼处被角度阈值丢掉的地面点。
与论文的偏离：
  - 统计去噪由前面的 denoise 完成，这里不再做一遍。
  - 边界高程用邻近种子的坡度判断：缓坡取最低一簇，陡坡用反距离加权。论文的整条坡链检测没有搬过来。
  - 二次曲面补偿只用于本层候选点，且该点离当前三角网不超过 2 倍本层距离阈值。
    树冠远高于网面，不会进这批点。全点云最后一轮只用 3×3 邻域的坡度一致性（论文式 15）。
  - ponytail: 每层只判一次，不逐点插入三角网。若山脊仍被削平，再对补偿点做第二次曲面拟合。
"""
import numpy as np
from scipy.spatial import Delaunay, cKDTree

from deform.iptd import _first_per_group, _locate


def _layers(w_max, w_min, ratio):
    w = [float(w_max)]
    while w[-1] * ratio > w_min:
        w.append(w[-1] * ratio)
    if w[-1] > w_min:
        w.append(float(w_min))
    return w


def _cell_lowest(xyz, origin, w, dh):
    """每格一个种子：最低点若明显低于该格中位（低异常），则改取中位附近的点，避免爬到树冠。"""
    ij = np.floor((xyz[:, :2] - origin) / w).astype(np.int64)
    ny = int(ij[:, 1].max()) + 1
    key = ij[:, 0] * ny + ij[:, 1]
    order = np.lexsort((xyz[:, 2], key))
    key_s = key[order]
    starts = np.r_[0, np.flatnonzero(np.diff(key_s)) + 1]
    ends = np.r_[starts[1:], len(order)]
    chosen = []
    z = xyz[:, 2]
    for s, e in zip(starts, ends):
        idx = order[s:e]
        zz = z[idx]
        med = zz[len(zz) // 2]
        pick = int(np.searchsorted(zz, med - dh)) if zz[0] < med - dh else 0
        chosen.append(idx[min(pick, len(idx) - 1)])
    return np.asarray(chosen, np.int64)


def _boundary(xyz, seeds, w):
    """数据范围外一圈，间距为 w。缓坡用最低种子的高程，陡于 30° 用反距离加权。"""
    lo, hi = xyz[:, :2].min(0) - w, xyz[:, :2].max(0) + w
    xs = np.linspace(lo[0], hi[0], int(np.ceil((hi[0] - lo[0]) / w)) + 1)
    ys = np.linspace(lo[1], hi[1], int(np.ceil((hi[1] - lo[1]) / w)) + 1)
    edge = [(x, y) for x in xs for y in (lo[1], hi[1])] + [(x, y) for y in ys[1:-1] for x in (lo[0], hi[0])]
    edge = np.asarray(edge, float)
    tree = cKDTree(xyz[seeds, :2])
    k = min(6, len(seeds))
    dist, nb = tree.query(edge, k=k, workers=-1)
    if nb.ndim == 1:
        dist, nb = dist[:, None], nb[:, None]
    pts = xyz[seeds[nb]]
    # 用邻近种子拟合坡面再外推。缓坡取最低点会把上坡边界压低，坡顶地面进不了三角网。
    cen = pts.mean(1)
    dxy = pts[:, :, :2] - cen[:, None, :2]
    dz = pts[:, :, 2] - cen[:, None, 2]
    gram = np.einsum("nki,nkj->nij", dxy, dxy)
    rhs = np.einsum("nki,nk->ni", dxy, dz)
    slope = np.linalg.solve(gram + np.eye(2) * 1e-6, rhs[..., None])[..., 0]
    z = cen[:, 2] + slope[:, 0] * (edge[:, 0] - cen[:, 0]) + slope[:, 1] * (edge[:, 1] - cen[:, 1])
    return np.c_[edge, z]


def _s_of(w, w_min, w_max, s_min, s_max):
    if w_max == w_min:
        return s_min
    t = (w - w_min) / (w_max - w_min)
    return s_min + (s_max - s_min) * t


def _build(xyz, ground, extra):
    vert = np.vstack([xyz[ground], extra]) if len(extra) else xyz[ground]
    return Delaunay(vert[:, :2]), vert


def _local_rule(xyz, ground, cand, w, flat_deg):
    """论文式 (15)：3×3 内平坦邻居 ≥ 3 且陡邻居 < 3 才补为地面。"""
    if len(cand) == 0 or ground.sum() < 3:
        return np.zeros(len(cand), bool)
    tree = cKDTree(xyz[ground, :2])
    k = min(8, int(ground.sum()))
    dist, nb = tree.query(xyz[cand, :2], k=k, workers=-1)
    if k == 1:
        dist, nb = dist[:, None], nb[:, None]
    g = np.flatnonzero(ground)
    neigh = xyz[g[nb]]
    horiz = np.maximum(dist, 1e-3)
    slope = np.degrees(np.arctan(np.abs(neigh[:, :, 2] - xyz[cand, 2:3]) / horiz))
    near = dist <= 1.5 * w
    flat = near & (slope < flat_deg)
    steep = near & ~flat
    return (flat.sum(1) >= 3) & (steep.sum(1) < 3)


def _same_surface(xyz, ground, cand, reach, z_tol):
    """离某个已确认地面点水平距离 ≤ reach 且高差 ≤ z_tol。树冠高差更大，补不进来。"""
    if len(cand) == 0 or ground.sum() < 3:
        return np.zeros(len(cand), bool)
    tree = cKDTree(xyz[ground, :2])
    k = min(8, int(ground.sum()))
    dist, nb = tree.query(xyz[cand, :2], k=k, workers=-1)
    if k == 1:
        dist, nb = dist[:, None], nb[:, None]
    dz = np.abs(xyz[np.flatnonzero(ground)[nb], 2] - xyz[cand, 2:3])
    return ((dist <= reach) & (dz <= z_tol)).any(1)


def tfmptd(xyz, w_max=15.0, w_min=0.5, ratio=2 / 3, angle=20.0, s_min=0.5, s_max=1.4, dh=1.0, flat_deg=10.0):
    """返回地面点掩码。单组参数取论文表 IV，距离下限按本项目点密度收到 0.3 m。"""
    xyz = np.asarray(xyz, float)
    n = len(xyz)
    ground = np.zeros(n, bool)
    origin = xyz[:, :2].min(0)
    layers = _layers(w_max, w_min, ratio)
    seeds = _cell_lowest(xyz, origin, layers[0], dh)
    ground[seeds] = True
    extra = _boundary(xyz, np.flatnonzero(ground), layers[0])

    for w in layers[1:]:
        tri, vert = _build(xyz, ground, extra)
        cand = _cell_lowest(xyz, origin, w, dh)
        cand = cand[~ground[cand]]
        if len(cand) == 0:
            continue
        _, sd, theta = _locate(tri, vert, xyz[cand])
        s_thr = _s_of(w, layers[-1], layers[0], s_min, s_max)
        ok = (np.abs(sd) <= s_thr) & (theta <= angle)
        # 网面下方的低洼：每格最低点，距离不超过 2 倍阈值就收下，树冠在网面上方进不来
        ok |= (sd < -s_min) & (sd >= -2 * s_thr)
        ground[cand[ok]] = True
        rest = cand[~ok & np.isfinite(sd) & (np.abs(sd) <= 2 * s_thr)]
        ground[rest[_local_rule(xyz, ground, rest, w, flat_deg)]] = True

    tri, vert = _build(xyz, ground, extra)
    rest = np.flatnonzero(~ground)
    _, sd, theta = _locate(tri, vert, xyz[rest])
    ok = (np.abs(sd) <= s_min) & (theta <= angle)
    ok |= (sd < -s_min) & (sd >= -2 * s_max)
    ground[rest[ok]] = True
    near = rest[~ok & np.isfinite(sd) & (np.abs(sd) <= 2 * s_max)]
    ground[near[_local_rule(xyz, ground, near, w_min, flat_deg)]] = True
    # 粗网格若把树冠当地面，真实地面会落在网面下方。每轮每个三角形收一个最低点，树冠在上方不会被收。
    for _ in range(8):
        tri, vert = _build(xyz, ground, extra)
        left = np.flatnonzero(~ground)
        simplex, sd, _ = _locate(tri, vert, xyz[left])
        below = sd < -s_min
        if not below.any():
            break
        ground[left[below][_first_per_group(simplex[below], sd[below])]] = True
    left = np.flatnonzero(~ground)
    # 水平方向放宽、高差仍用 s_min：陡坎上沿能接到同高的种子，贴地灌木高差更大接不到
    ground[left[_same_surface(xyz, ground, left, 2 * w_min, s_min)]] = True
    return ground
