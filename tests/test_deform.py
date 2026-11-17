"""合成数据自检：带坡、陡坎和模拟植被的地形，逐模块验证。运行：python -m pytest tests/test_deform.py"""
import json
from pathlib import Path

import numpy as np

SIZE = (60.0, 40.0)


def terrain(x, y):
    """约 27 度的斜坡 + 起伏 + x=30 处 3 m 高的陡坎。"""
    return 0.5 * x + 2.0 * np.sin(y / 8.0) + np.where(x > 30.0, 3.0, 0.0)


def make_scene(seed=0, density=40.0, noise=0.01):
    """返回 (xyz, is_ground)。树下地面点只保留 30%，模拟林下稀疏。"""
    rng = np.random.default_rng(seed)
    n = int(SIZE[0] * SIZE[1] * density)
    xy = rng.uniform((0, 0), SIZE, (n, 2))
    trees = rng.uniform((5, 5), (SIZE[0] - 5, SIZE[1] - 5), (12, 2))
    under = (np.linalg.norm(xy[:, None] - trees[None], axis=2) < 3.0).any(1)
    xy = xy[~under | (rng.random(n) < 0.3)]
    ground = np.c_[xy, terrain(*xy.T) + rng.normal(0, noise, len(xy))]
    veg = []
    for t in trees:
        r, a = 3.0 * np.sqrt(rng.random(1500)), rng.uniform(0, 2 * np.pi, 1500)
        p = t + np.c_[r * np.cos(a), r * np.sin(a)]
        veg.append(np.c_[p, terrain(*p.T) + rng.uniform(2.0, 8.0, 1500)])
    p = rng.uniform((0, 0), SIZE, (4000, 2))
    veg.append(np.c_[p, terrain(*p.T) + rng.uniform(0.3, 1.5, 4000)])
    veg = np.vstack(veg)
    return np.vstack([ground, veg]), np.r_[np.ones(len(ground), bool), np.zeros(len(veg), bool)]


def make_ground_only(seed=0, density=30.0, noise=0.005):
    rng = np.random.default_rng(seed)
    n = int(SIZE[0] * SIZE[1] * density)
    xy = rng.uniform((0, 0), SIZE, (n, 2))
    return np.c_[xy, terrain(*xy.T) + rng.normal(0, noise, n)]


def test_denoise_removes_isolated_outliers_and_keeps_sparse_ground():
    from deform.denoise import denoise

    xyz, is_ground = make_scene()
    rng = np.random.default_rng(1)
    xy = rng.uniform((0, 0), SIZE, (300, 2))
    outliers = np.c_[xy, terrain(*xy.T) + rng.choice([-1, 1], 300) * rng.uniform(15, 40, 300)]

    keep = denoise(np.vstack([xyz, outliers]))

    assert keep[len(xyz):].mean() <= 0.05, "注入的离群点删除率应 >= 95%"
    assert keep[: len(xyz)][is_ground].mean() >= 0.99, "地面点（含林下稀疏区）保留率应 >= 99%"


def test_iptd_separates_ground_from_vegetation_and_keeps_scarp():
    from deform.iptd import iptd

    xyz, is_ground = make_scene()
    ground = iptd(xyz)

    assert (~ground[is_ground]).mean() < 0.05, "地面点被错判为非地面的比例应 < 5%"
    assert ground[~is_ground].mean() < 0.05, "非地面点被错判为地面的比例应 < 5%"
    near_scarp = is_ground & (np.abs(xyz[:, 0] - 30.0) < 1.0)
    assert ground[near_scarp].mean() > 0.8, "陡坎上下 1 m 内的地面点应大部分保留"


def test_register_recovers_known_shift_with_deformation_patch():
    from deform.register import apply_transform, register

    ref = make_ground_only(seed=2)
    shift = np.array([0.05, -0.03, 0.02])
    tgt = ref + shift
    # 一块抬升 20 cm 的形变，不应把配准带偏太多
    patch = (tgt[:, 0] > 40) & (tgt[:, 0] < 50) & (tgt[:, 1] > 10) & (tgt[:, 1] < 25)
    tgt = tgt.copy()
    tgt[patch, 2] += 0.20

    matrix, info = register(ref, tgt, reduction_point=np.mean(ref, 0), voxel=0.5,
                            supervoxel_resolution=3.0, initial_distance_threshold=0.3,
                            level_of_detection=0.05, max_translation=0.5, max_rotation_deg=1.0)
    aligned = apply_transform(tgt, matrix, np.mean(ref, 0))
    # 稳定区（非 patch）上的中位位移应接近 0
    stable = ~patch
    med = np.median(aligned[stable] - ref[stable], axis=0)
    assert np.linalg.norm(med) < 0.03, f"稳定区残余位移过大: {med}, status={info['status']}"


def test_m3c2_recovers_10cm_uplift_and_colorize_sign():
    from deform.change import colorize, m3c2, voxel_corepoints

    ref = make_ground_only(seed=3, density=40.0)
    tgt = ref.copy()
    patch = (tgt[:, 0] > 35) & (tgt[:, 0] < 45) & (tgt[:, 1] > 12) & (tgt[:, 1] < 22)
    tgt[patch, 2] += 0.10

    core = voxel_corepoints(tgt, 0.5)
    deformation, lod, significant, reg, _ = m3c2(ref, tgt, core, normal_radii=[1.0], cyl_radius=0.75, max_distance=2.0)
    in_patch = ((core[:, 0] > 35) & (core[:, 0] < 45) & (core[:, 1] > 12) & (core[:, 1] < 22))
    assert np.nanmedian(deformation[in_patch]) > 0.05, "抬升块形变应为正且接近 0.10 m"
    assert np.nanmedian(deformation[in_patch]) < 0.15
    # 稳定区显著比例不应过高
    assert significant[~in_patch].mean() < 0.2

    rgb = colorize(deformation, significant, vmax=0.15)
    pos = significant & (deformation > 0)
    if pos.any():
        assert rgb[pos, 0].mean() > rgb[pos, 2].mean(), "正值应偏暖色（红 > 蓝）"


def _write_xyz_las(path, xyz):
    import laspy

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor(xyz.min(0))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.write(str(path))


def test_tfmptd_rejects_canopy_and_keeps_scarp():
    from deform.tfmptd import tfmptd

    xyz, is_ground = make_scene()
    ground = tfmptd(xyz)
    above = xyz[:, 2] - terrain(xyz[:, 0], xyz[:, 1])

    assert (~ground[is_ground]).mean() < 0.02, "地面点被错判为非地面的比例应 < 2%"
    assert ground[(~is_ground) & (above > 2.0)].mean() < 0.01, "高于地面 2 m 的树冠不应被当成地面"
    near_scarp = is_ground & (np.abs(xyz[:, 0] - 30.0) < 1.0)
    assert ground[near_scarp].mean() > 0.7, "陡坎上下 1 m 内的地面点应大部分保留"


def test_colorbar_marks_lod_median(tmp_path):
    from deform.change import write_colorbar

    path = tmp_path / "colorbar.png"
    write_colorbar(path, vmax=5.73, lod_median=0.186)
    assert path.stat().st_size > 1000


def test_pipeline_end_to_end(tmp_path):
    from deform.pipeline import run

    # 小场景但保证 RoI 内地面点 > 1000：用较高密度、较小缓冲
    ref = make_ground_only(seed=4, density=50.0)
    tgt = ref.copy()
    patch = (tgt[:, 0] > 35) & (tgt[:, 0] < 45) & (tgt[:, 1] > 12) & (tgt[:, 1] < 22)
    tgt[patch, 2] += 0.12
    # 加少量植被，避免 IPTD 把一切当地面
    rng = np.random.default_rng(4)
    veg_xy = rng.uniform((5, 5), (55, 35), (3000, 2))
    veg = np.c_[veg_xy, terrain(*veg_xy.T) + rng.uniform(2.0, 6.0, 3000)]
    ref2 = np.vstack([ref, veg])
    tgt2 = np.vstack([tgt, veg])

    base = tmp_path / "base.las"
    target = tmp_path / "target.las"
    _write_xyz_las(base, ref2)
    _write_xyz_las(target, tgt2)

    roi = {
        "target_id": "syn_A",
        "center": [30.0, 20.0, float(np.median(ref[:, 2]))],
        "center_axis": [0.0, 0.0, 1.0],
        "w": 50.0,
        "h": 30.0,
        "l": 40.0,
    }
    roi_path = tmp_path / "roi.json"
    roi_path.write_text(json.dumps(roi), encoding="utf-8")
    out = tmp_path / "out"

    # 合成场景缓冲带可以小一些；IPTD 用默认
    results = run(
        str(base),
        str(target),
        str(roi_path),
        str(out),
        buffer=5.0,
        params={"register": {"supervoxel_resolution": 3.0, "max_rotation_deg": 1.0}},
        save_stages=True,
    )
    assert len(results) == 1
    d = out / "syn_A"
    for name in ("ground_heatmap.las", "roi_heatmap.las", "colorbar.png", "result.json"):
        assert (d / name).exists(), name
    result = json.loads((d / "result.json").read_text(encoding="utf-8"))
    assert "counts" in result and result["counts"]["corepoints"] > 0
    assert result["registration"]["status"] in ("applied", "rejected", "fallback")
    for name in result["stages"]:
        assert (d / name).exists(), name
