"""IPTD：改进的渐进式不规则三角网加密地面滤波（Zhao 等 2016，ISPRS J. 117: 79–91）。

三步：形态学开运算取候选种子点 → 平移平面拟合筛选种子点 → 三角网先向下、再向上迭代加密。
与论文的偏离（均因 L2 点密度高、纯 Python 实现）：
  - 栅格与阈值按点密度缩小（论文数据约 10 pts/m²，栅格 1 m）。
  - 向上加密分轮整批进行：每轮每个三角形只接收最低的一个合格点，然后重建三角网。
  - 最长边小于 min_edge 的三角形不再加密；结束后用最终三角网对剩余点判定一次。
  - 到三角面的距离在噪声量级（noise）以内的点直接接受，不看角度：
    点离顶点很近时，厘米级噪声就会产生很大的夹角。
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree


def _first_per_group(group, value):
    """按 group 分组，返回每组 value 最小的元素下标。"""
    order = np.lexsort((value, group))
    g = group[order]
    return order[np.r_[True, g[1:] != g[:-1]]]


def _seeds(xyz, cell, window, seed_tol, k, plane_rms):
    """论文 3.1–3.2：返回地面种子点在 xyz 中的下标。"""
    ij = np.floor((xyz[:, :2] - xyz[:, :2].min(0)) / cell).astype(np.int64)
    nx, ny = ij.max(0) + 1
    label = ij[:, 1] * nx + ij[:, 0]
    lowest = _first_per_group(label, xyz[:, 2])  # 每格最低点
    grid = np.full(ny * nx, np.nan)
    grid[label[lowest]] = xyz[lowest, 2]
    grid = grid.reshape(ny, nx)
    empty = np.isnan(grid)
    if empty.any():  # 空格用最近有值格填充（论文对大空洞用边界最低值，这里简化）
        nearest = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
        grid = grid[tuple(nearest)]
    size = int(round(window / cell)) | 1
    opened = ndimage.grey_opening(grid, size=(size, size))
    cand = lowest[np.abs(xyz[lowest, 2] - opened.ravel()[label[lowest]]) < seed_tol]

    # 平移平面拟合：近邻拟合平面，平移到过该点，近邻到平移平面的均方根距离过大则剔除
    p = xyz[cand]
    _, nb = cKDTree(p).query(p, k=min(k + 1, len(p)), workers=-1)
    q = p[nb]  # (n, k+1, 3)，第 0 个是自身
    c = q - q.mean(1, keepdims=True)
    normal = np.linalg.eigh(np.einsum("nki,nkj->nij", c, c))[1][:, :, 0]  # 最小特征值的特征向量
    d = np.einsum("nki,ni->nk", q[:, 1:] - p[:, None], normal)
    return cand[np.sqrt((d ** 2).mean(1)) <= plane_rms]


def _locate(tri, vert, pts, chunk=1_000_000):
    """对每个点返回：所在三角形号、到三角面的带符号垂距（上正下负）、与最近顶点连线同三角面的夹角（度）。"""
    n = len(pts)
    simplex, sd, theta = np.empty(n, np.int64), np.empty(n), np.empty(n)
    for a in range(0, n, chunk):
        p = pts[a:a + chunk]
        s = tri.find_simplex(p[:, :2])
        v = vert[tri.simplices[s]]  # (m, 3, 3)
        normal = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        normal *= np.sign(normal[:, 2:3])
        d = np.einsum("ni,ni->n", p - v[:, 0], normal)
        nearest = np.linalg.norm(p[:, None] - v, axis=2).min(1)
        t = np.degrees(np.arcsin(np.clip(np.abs(d) / np.maximum(nearest, 1e-9), 0, 1)))
        d[s < 0] = np.inf  # 落在三角网外（不应发生，有缓冲区模拟点）
        simplex[a:a + chunk], sd[a:a + chunk], theta[a:a + chunk] = s, d, t
    return simplex, sd, theta


def iptd(xyz, cell=0.5, window=3.0, seed_tol=0.5, k=15, plane_rms=0.5, angle=20.0, dist=0.3,
         buffer=30.0, noise=0.05, min_edge=0.2, max_iter=30):
    """返回地面点掩码。单位为米和度；参数含义见模块说明与论文。"""
    xyz = np.asarray(xyz, float)
    ground = np.zeros(len(xyz), bool)
    seeds = _seeds(xyz, cell, window, seed_tol, k, plane_rms)
    ground[seeds] = True

    # 论文 3.3.1：缓冲区边界上的模拟地面点，高程取最近种子点，避免边缘出现狭长三角形
    lo, hi = xyz[:, :2].min(0) - buffer, xyz[:, :2].max(0) + buffer
    xs = np.linspace(lo[0], hi[0], int(np.ceil((hi[0] - lo[0]) / buffer)) + 1)
    ys = np.linspace(lo[1], hi[1], int(np.ceil((hi[1] - lo[1]) / buffer)) + 1)
    border = np.array([(x, y) for x in xs for y in (lo[1], hi[1])] + [(x, y) for y in ys[1:-1] for x in (lo[0], hi[0])])
    simulated = np.c_[border, xyz[seeds[cKDTree(xyz[seeds, :2]).query(border)[1]], 2]]

    def build():
        vert = np.vstack([xyz[ground], simulated])
        return Delaunay(vert[:, :2]), vert

    # 论文 3.3.2 向下加密：每个三角形下方最远的点并入地面
    for _ in range(5):
        tri, vert = build()
        rest = np.flatnonzero(~ground)
        simplex, sd, _ = _locate(tri, vert, xyz[rest])
        below = sd < -noise
        if not below.any():
            break
        ground[rest[below][_first_per_group(simplex[below], sd[below])]] = True

    # 论文 3.3.3 向上加密
    def accepted(final):
        tri, vert = build()
        rest = np.flatnonzero(~ground)
        simplex, sd, theta = _locate(tri, vert, xyz[rest])
        ok = (np.abs(sd) <= dist) & ((theta <= angle) | (np.abs(sd) <= noise))
        if final:
            return rest[ok]
        v = vert[tri.simplices][:, :, :2]
        edge = np.linalg.norm(v - np.roll(v, 1, axis=1), axis=2).max(1)  # 每个三角形的最长边
        ok &= edge[simplex] >= min_edge
        return rest[ok][_first_per_group(simplex[ok], xyz[rest[ok], 2])]  # 每个三角形只取最低的一个

    for _ in range(max_iter):
        add = accepted(final=False)
        ground[add] = True
        if len(add) < 0.001 * ground.sum():
            break
    ground[accepted(final=True)] = True
    return ground
