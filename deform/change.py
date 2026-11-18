"""M3C2 形变识别、LoD 显著性、热图着色与色标图。"""
import numpy as np
import py4dgeo
from scipy.spatial import cKDTree


def voxel_corepoints(xyz, voxel=0.25):
    """每个平面格保留第一个点，作为 M3C2 核心点。"""
    if len(xyz) == 0:
        return xyz
    ij = np.floor((xyz[:, :2] - xyz[:, :2].min(0)) / voxel).astype(np.int64)
    _, idx = np.unique(ij, axis=0, return_index=True)
    return xyz[np.sort(idx)]


def stable_registration_error(deformation):
    """配准误差只从形变最集中的那一半点上估计。

    这半点是数值上的稳定点，不要求落在某一块地面上。开挖、残留植被等大形变
    落在这段窗口外面，不参与。正态总体下，含一半样本的最短区间宽度约为 1.349σ。
    """
    x = np.sort(np.asarray(deformation, float))
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return 0.0
    k = max(n // 2, 2)
    width = x[k - 1:] - x[:n - k + 1]
    return float(np.min(width) / 1.349)


def m3c2(
    ref_ground,
    tgt_ground,
    corepoints,
    normal_radii=(0.5, 1.0, 2.0),
    cyl_radius=0.5,
    max_distance=10.0,
    min_samples=5,
):
    """在核心点上计算形变。

    epochs 顺序为 (目标期, 基准期)，法线取自目标期；形变 = -distances，
    使正值表示目标期相对基准期沿法线向外/向上。

    返回 deformation, lod, significant, registration_error, uncertainties。
    """
    ref = py4dgeo.Epoch(np.ascontiguousarray(ref_ground, np.float64))
    tgt = py4dgeo.Epoch(np.ascontiguousarray(tgt_ground, np.float64))
    core = np.ascontiguousarray(corepoints, np.float64)

    algo = py4dgeo.M3C2(
        epochs=(tgt, ref),
        corepoints=core,
        normal_radii=list(normal_radii),
        cyl_radius=cyl_radius,
        max_distance=max_distance,
        registration_error=0.0,
        orientation_vector=np.array([0.0, 0.0, 1.0]),
    )
    distances, unc = algo.run()
    deformation = -np.asarray(distances, float)

    reg = stable_registration_error(deformation)
    valid = np.isfinite(deformation)

    lod0 = np.asarray(unc["lodetection"], float)
    lod = lod0 + 1.96 * reg
    n1 = np.asarray(unc["num_samples1"])
    n2 = np.asarray(unc["num_samples2"])
    significant = valid & (np.abs(deformation) > lod) & (n1 >= min_samples) & (n2 >= min_samples)
    return deformation, lod, significant, reg, unc


def propagate_to_points(points, corepoints, deformation, lod, significant, max_dist=0.5):
    """把核心点结果传到全部地面点；超过 max_dist 记为无数据（nan / False）。"""
    n = len(points)
    out_d = np.full(n, np.nan)
    out_l = np.full(n, np.nan)
    out_s = np.zeros(n, bool)
    if len(corepoints) == 0 or n == 0:
        return out_d, out_l, out_s
    dist, idx = cKDTree(corepoints).query(points, workers=-1)
    ok = dist <= max_dist
    out_d[ok] = deformation[idx[ok]]
    out_l[ok] = lod[idx[ok]]
    out_s[ok] = significant[idx[ok]]
    return out_d, out_l, out_s


def choose_vmax(deformation, significant, vmin=0.10):
    """默认取显著形变绝对值的 98% 分位，且不小于 vmin。"""
    v = np.abs(deformation[significant & np.isfinite(deformation)])
    if len(v) == 0:
        return float(vmin)
    return float(max(vmin, np.percentile(v, 98)))


def colorize(deformation, significant, vmax, no_data_mask=None):
    """返回 uint8 RGB。显著：RdBu_r；不显著白；无数据深灰。"""
    import matplotlib.cm as cm

    n = len(deformation)
    rgb = np.full((n, 3), 255, np.uint8)  # 默认白 = 不显著
    if no_data_mask is None:
        no_data_mask = ~np.isfinite(deformation)
    rgb[no_data_mask] = (80, 80, 80)

    mask = significant & np.isfinite(deformation) & ~no_data_mask
    if mask.any() and vmax > 0:
        t = np.clip(deformation[mask] / vmax, -1.0, 1.0)
        try:
            cmap = __import__("matplotlib").colormaps["RdBu_r"]
        except (AttributeError, KeyError):
            cmap = cm.get_cmap("RdBu_r")
        colors = np.asarray(cmap((t + 1.0) * 0.5))[:, :3]
        rgb[mask] = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    return rgb


def write_colorbar(path, vmax, lod_median=None):
    """写出距离-颜色对应条。冷暖从各点自己的检测限开始；虚线标出检测限中位数。"""
    import os
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import Normalize

    font = None
    for name in ("msyh.ttc", "simhei.ttf"):
        fp = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
        if os.path.isfile(fp):
            font = font_manager.FontProperties(fname=fp)
            break

    fig, ax = plt.subplots(figsize=(8, 1.8))
    gradient = np.linspace(-vmax, vmax, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap="RdBu_r", norm=Normalize(-vmax, vmax), extent=[-vmax, vmax, 0, 1])
    ax.set_yticks([])
    ax.set_xlabel("沿法线的形变（米）：暖色上升，冷色沉降", fontproperties=font)
    if lod_median is not None and np.isfinite(lod_median) and lod_median < vmax:
        for x in (-lod_median, lod_median):
            ax.axvline(x, color="k", lw=0.8, ls="--")
        ax.set_title(
            f"虚线为检测限中位数 {lod_median:.3f} m\n",
            # f"颜色在 ±{vmax:.3f} m 饱和",
            fontproperties=font,
        )
    else:
        # ax.set_title(f"颜色在 ±{vmax:.3f} m 饱和", fontproperties=font)
        ax.set_title(f"检测限 {lod_median:.3f} m 为空值", fontproperties=font)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def extrema(corepoints, deformation, lod, significant):
    """返回 max_positive / max_negative 字典或 None。"""
    out = {"max_positive": None, "max_negative": None}
    m = significant & np.isfinite(deformation)
    if not m.any():
        return out
    d, c, l = deformation[m], corepoints[m], lod[m]
    i = int(np.argmax(d))
    if d[i] > 0:
        out["max_positive"] = {"value": float(d[i]), "xyz": c[i].tolist(), "lod": float(l[i])}
    j = int(np.argmin(d))
    if d[j] < 0:
        out["max_negative"] = {"value": float(d[j]), "xyz": c[j].tolist(), "lod": float(l[j])}
    return out
