"""布料模拟滤波，并补上陡坡/台阶、剥掉贴地植被。

只调一个距离阈值做不到这两件事：阈值小，阶梯和陡坡离布料远，会被整片删掉；阈值大，矮植被留在地面里，后面会被当成形变。
这里分三步：
  1. 软布料（硬度 1）加陡坡后处理，分类距离放宽，先别删坡。
  2. 布料没留住、但贴着已保留地面三角网的点补回（陡坎、台阶侧面）。
  3. 用每格最低点铺一张地面网，明显高于这张网的点当作矮植被去掉。
格里如果根本没有地面回波，最低点仍是植被，这一步去不掉。
"""
import numpy as np
from scipy.spatial import Delaunay

from deform.iptd import _locate


def _csf(xyz, cloth_resolution, rigidness, time_step, class_threshold, iterations, slope_smooth):
    import CSF

    csf = CSF.CSF()
    p = csf.params
    p.bSloopSmooth = bool(slope_smooth)
    p.cloth_resolution = float(cloth_resolution)
    p.rigidness = int(rigidness)
    p.time_step = float(time_step)
    p.class_threshold = float(class_threshold)
    p.interations = int(iterations)
    csf.setPointCloud(np.ascontiguousarray(xyz, np.float64))
    ground, off = CSF.VecInt(), CSF.VecInt()
    csf.do_filtering(ground, off)
    mask = np.zeros(len(xyz), bool)
    mask[np.asarray(list(ground), np.int64)] = True
    return mask


def _anchors(xyz, mask, cell):
    """每个平面格里最低的地面点，用来代表局部地面，而不是植被顶。"""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return idx
    ij = np.floor((xyz[idx, :2] - xyz[idx, :2].min(0)) / cell).astype(np.int64)
    key = ij[:, 0] * (int(ij[:, 1].max()) + 1) + ij[:, 1]
    order = np.lexsort((xyz[idx, 2], key))
    key_s = key[order]
    first = np.r_[True, key_s[1:] != key_s[:-1]]
    return idx[order[first]]


def _above_tin(xyz, anchors, query):
    """query 各点高出锚点三角网多少米。网外为 inf，低于网面为负。"""
    if len(anchors) < 3 or len(query) == 0:
        return np.zeros(len(query))
    extra_xy = np.array([
        xyz[:, :2].min(0) - 1.0,
        [xyz[:, 0].min() - 1.0, xyz[:, 1].max() + 1.0],
        [xyz[:, 0].max() + 1.0, xyz[:, 1].min() - 1.0],
        xyz[:, :2].max(0) + 1.0,
    ])
    near = anchors[np.argmin(np.linalg.norm(xyz[anchors, :2][:, None] - extra_xy, axis=2), axis=0)]
    vert = np.vstack([xyz[anchors], np.c_[extra_xy, xyz[near, 2]]])
    tri = Delaunay(vert[:, :2])
    _, sd, _ = _locate(tri, vert, xyz[query])
    return sd


def csf_ground(xyz, cloth_resolution=0.5, rigidness=1, time_step=0.65, class_threshold=0.5,
               iterations=500, slope_smooth=True, cell=0.5, veg_cut=0.3, restore_dist=0.4):
    """返回地面点掩码。veg_cut 是允许高出局部最低地面的高度，再高就当矮植被。"""
    xyz = np.asarray(xyz, float)
    mask = _csf(xyz, cloth_resolution, rigidness, time_step, class_threshold, iterations, slope_smooth)
    anchors = _anchors(xyz, mask, cell)
    if len(anchors) < 3:
        return mask
    dropped = np.flatnonzero(~mask)
    sd = _above_tin(xyz, anchors, dropped)
    mask[dropped[(np.abs(sd) <= restore_dist)]] = True
    anchors = _anchors(xyz, mask, cell)
    kept = np.flatnonzero(mask)
    sd = _above_tin(xyz, anchors, kept)
    mask[kept[sd > veg_cut]] = False
    return mask
