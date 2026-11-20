"""合成地形上测 M3C2 精度和耗时。不进 pytest。

固定：60 m × 40 m，项目里的斜坡+起伏+陡坎；两期独立采样；
三维高斯噪声 σ = 1 cm；x∈(34,56)、y∈(6,34) 的一块沿解析法线抬升 10 cm。
统计只用地块内部（再缩进 3 m，大于最大法线半径 2 m）。
核心点网格、圆柱半径、法线半径用 deform.change.m3c2 的默认值。
变的只有点密度。陡坎两侧 1 m、地块边界 2 m 不计入精度。

运行：python tests/bench_m3c2.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_deform import SIZE, _write_xyz_las, terrain  # noqa: E402

from deform.change import m3c2, voxel_corepoints  # noqa: E402

NOISE = 0.01
TRUE = 0.10
DENSITIES = (10.0, 40.0, 100.0)
SEEDS = (0, 1, 2)
PATCH = (34.0, 56.0, 6.0, 34.0)  # 避开 x=30 陡坎；内部再缩 3 m，大于最大法线半径 2 m
INSET = 3.0


def upward_normal(x, y):
    gx = np.full_like(x, 0.5, dtype=float)
    gy = 0.25 * np.cos(np.asarray(y, float))
    n = np.column_stack([-gx, -gy, np.ones_like(gx)])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def sample(seed, density, deform):
    rng = np.random.default_rng(seed)
    n = int(SIZE[0] * SIZE[1] * density)
    xy = rng.uniform((0, 0), SIZE, (n, 2))
    xyz = np.column_stack([xy, terrain(xy[:, 0], xy[:, 1])])
    if deform:
        x0, x1, y0, y1 = PATCH
        m = (xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)
        xyz[m] += TRUE * upward_normal(xy[m, 0], xy[m, 1])
    xyz += rng.normal(0, NOISE, xyz.shape)
    return xyz


def _inside(xy, box, inset):
    x0, x1, y0, y1 = box
    return (xy[:, 0] > x0 + inset) & (xy[:, 0] < x1 - inset) & (xy[:, 1] > y0 + inset) & (xy[:, 1] < y1 - inset)


def stats(meas, sig, truth):
    finite = np.isfinite(meas)
    n = len(meas)
    if not finite.any():
        return dict(n=n, finite=0, median=np.nan, bias=np.nan, robust=np.nan, rmse=np.nan, rate=0.0)
    err = meas[finite] - truth
    bias = float(np.median(err))
    robust = float(1.4826 * np.median(np.abs(err - bias)))
    return dict(
        n=n,
        finite=int(finite.sum()),
        median=float(np.median(meas[finite])),
        bias=bias,
        robust=robust,
        rmse=float(np.sqrt(np.mean(err ** 2))),
        rate=float(sig.mean()) if n else np.nan,
    )


def regions(core):
    xy = core[:, :2]
    patch = _inside(xy, PATCH, INSET)
    x0, x1, y0, y1 = PATCH
    near = (xy[:, 0] > x0 - INSET) & (xy[:, 0] < x1 + INSET) & (xy[:, 1] > y0 - INSET) & (xy[:, 1] < y1 + INSET)
    border = (xy[:, 0] < INSET) | (xy[:, 0] > SIZE[0] - INSET) | (xy[:, 1] < INSET) | (xy[:, 1] > SIZE[1] - INSET)
    scarp = np.abs(xy[:, 0] - 30.0) < 1.0
    stable = ~near & ~border & ~scarp
    return patch, stable


def run_m3c2(density, seed):
    ref = sample(seed, density, deform=False)
    tgt = sample(seed + 1000, density, deform=True)
    core = voxel_corepoints(tgt, 0.25)
    t0 = time.perf_counter()
    deformation, lod, significant, reg, unc = m3c2(ref, tgt, core)
    elapsed = time.perf_counter() - t0
    patch, stable = regions(core)
    return {
        "density": density,
        "seed": seed,
        "n": len(tgt),
        "core": len(core),
        "seconds": elapsed,
        "reg": reg,
        "lod": float(np.nanmedian(lod[np.isfinite(lod)])),
        "samples": float(np.median(np.minimum(unc["num_samples1"], unc["num_samples2"]))),
        "patch": stats(deformation[patch], significant[patch], TRUE),
        "stable": stats(deformation[stable], significant[stable], 0.0),
    }


def fmt(row):
    p, s = row["patch"], row["stable"]
    return (
        f"{row['density']:6.0f} {row['seed']:4d} {row['n']:8d} {row['core']:7d} {row['seconds']:7.2f} "
        f"{p['median']:7.4f} {p['bias']:+7.4f} {p['robust']:7.4f} {p['rmse']:7.4f} {p['rate']:6.3f} "
        f"{s['robust']:7.4f} {s['rate']:6.3f} {row['lod']:7.4f} {row['reg']:7.4f} {row['samples']:6.1f}"
    )


def mean_rows(rows):
    def avg(path):
        return float(np.mean([path(r) for r in rows]))

    return {
        "seconds": avg(lambda r: r["seconds"]),
        "bias": avg(lambda r: r["patch"]["bias"]),
        "robust": avg(lambda r: r["patch"]["robust"]),
        "rmse": avg(lambda r: r["patch"]["rmse"]),
        "detect": avg(lambda r: r["patch"]["rate"]),
        "stable_robust": avg(lambda r: r["stable"]["robust"]),
        "fpr": avg(lambda r: r["stable"]["rate"]),
        "lod": avg(lambda r: r["lod"]),
        "samples": avg(lambda r: r["samples"]),
    }


def run_pipeline(density, seed):
    from deform.pipeline import process_roi

    ref = sample(seed, density, deform=False)
    tgt = sample(seed + 1000, density, deform=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _write_xyz_las(tmp / "base.las", ref)
        _write_xyz_las(tmp / "target.las", tgt)
        roi = {
            "target_id": "bench",
            "center": [30.0, 20.0, float(np.median(ref[:, 2]))],
            "w": 50.0,
            "h": 30.0,
            "l": 40.0,
        }
        (tmp / "roi.json").write_text(__import__("json").dumps(roi), encoding="utf-8")
        t0 = time.perf_counter()
        result = process_roi(tmp / "base.las", tmp / "target.las", roi, tmp / "out", buffer=5.0)
        elapsed = time.perf_counter() - t0
        reg = result["registration"]
        import laspy

        las = laspy.read(tmp / "out" / "bench" / "ground_heatmap.las")
        xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
        meas = np.asarray(las.deformation, float)
        meas[np.asarray(las.lod) < 0] = np.nan
        sig = np.asarray(las.significant).astype(bool)
    patch, stable = regions(xyz)
    return {
        "seconds": elapsed,
        "timings": result["timings"],
        "status": reg["status"],
        "translation": reg["translation"],
        "warnings": result["warnings"],
        "counts": result["counts"],
        "patch": stats(meas[patch], sig[patch], TRUE),
        "stable": stats(meas[stable], sig[stable], 0.0),
    }


def main():
    n = upward_normal(np.array([10.0]), np.array([8.0]))
    assert abs(np.linalg.norm(n) - 1) < 1e-9 and n[0, 2] > 0
    print(
        f"固定：面积 {SIZE[0]:.0f}×{SIZE[1]:.0f} m，噪声 σ={NOISE} m，"
        f"地块沿法线 {TRUE} m，核心格 0.25 m，M3C2 默认参数。3 个种子。"
    )
    header = (
        "dens seed  n_pts   n_core   sec   median    bias  robust    rmse detect "
        "st_rob  st_fp    lod    reg  n_cyl"
    )
    print(header)
    by_density = {d: [] for d in DENSITIES}
    for density in DENSITIES:
        for seed in SEEDS:
            row = run_m3c2(density, seed)
            by_density[density].append(row)
            print(fmt(row), flush=True)
    print("\n三种子平均（M3C2，不含去噪/地面/配准）")
    print("dens   sec   bias  robust   rmse detect st_rob  st_fp    lod  n_cyl")
    for density, rows in by_density.items():
        a = mean_rows(rows)
        print(
            f"{density:4.0f} {a['seconds']:6.2f} {a['bias']:+7.4f} {a['robust']:7.4f} "
            f"{a['rmse']:7.4f} {a['detect']:6.3f} {a['stable_robust']:7.4f} {a['fpr']:6.3f} "
            f"{a['lod']:7.4f} {a['samples']:6.1f}"
        )
    print("\n整条流程一次：密度 40 pts/m²，种子 0，默认 IPTD + 配准 + M3C2")
    pipe = run_pipeline(40.0, 0)
    p, s = pipe["patch"], pipe["stable"]
    print(f"wall {pipe['seconds']:.2f} s  timings {pipe['timings']}")
    print(
        f"registration {pipe['status']}  translation {pipe['translation']}  warnings {pipe['warnings']}"
    )
    print(f"counts {pipe['counts']}")
    print(
        f"热图地面点 地块 median {p['median']:.4f} bias {p['bias']:+.4f} "
        f"robust {p['robust']:.4f} rmse {p['rmse']:.4f} detect {p['rate']:.3f}"
    )
    print(f"稳定区 robust {s['robust']:.4f} 误报 {s['rate']:.3f}")


if __name__ == "__main__":
    main()
