"""真实数据验证（手动运行，不进 pytest）:

python tests/validate_real.py
python tests/validate_real.py --full   # 跑完整流水线（耗时长）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import laspy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
BASE = DATA / "20251121.las"
TARGET = DATA / "20260911.las"
ROI = DATA / "mountain_roi.json"


def _clip_xyz(path, center, w, h, buffer, max_points=None):
    from deform.pipeline import read_clipped_xyz

    xyz = read_clipped_xyz(path, center, w, h, buffer)
    if max_points and len(xyz) > max_points:
        rng = np.random.default_rng(0)
        xyz = xyz[rng.choice(len(xyz), max_points, replace=False)]
    return xyz


def check_denoise_safety(center, w, h):
    """目标期自带地面点被删除比例应 < 1%。"""
    from deform.denoise import denoise

    # 只取已分类地面点做检查
    half = np.array([w / 2 + 20, h / 2 + 20])
    lo, hi = center[:2] - half, center[:2] + half
    parts = []
    with laspy.open(TARGET) as r:
        for pts in r.chunk_iterator(3_000_000):
            cls = np.asarray(pts.classification)
            x, y, z = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)
            m = (cls == 2) & (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1])
            if m.any():
                parts.append(np.column_stack([x[m], y[m], z[m]]))
    if not parts:
        print("[denoise] SKIP: target has no class-2 points in buffer")
        return
    xyz = np.vstack(parts)
    if len(xyz) > 300_000:
        xyz = xyz[np.random.default_rng(0).choice(len(xyz), 300_000, replace=False)]
    local = xyz - center
    keep = denoise(local)
    removed = 1.0 - keep.mean()
    ok = removed < 0.01
    print(f"[denoise] removed {removed:.3%} of class-2 points -> {'PASS' if ok else 'FAIL'} (<1%)")


def check_iptd_agreement(center, w, h):
    """与目标期自带地面分类的一致率。"""
    from deform.denoise import denoise
    from deform.iptd import iptd

    half = np.array([w / 2 + 20, h / 2 + 20])
    lo, hi = center[:2] - half, center[:2] + half
    xyz_parts, cls_parts = [], []
    with laspy.open(TARGET) as r:
        for pts in r.chunk_iterator(3_000_000):
            x, y, z = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)
            m = (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1])
            if m.any():
                xyz_parts.append(np.column_stack([x[m], y[m], z[m]]))
                cls_parts.append(np.asarray(pts.classification)[m])
    xyz = np.vstack(xyz_parts)
    cls = np.concatenate(cls_parts)
    if len(xyz) > 400_000:
        idx = np.random.default_rng(1).choice(len(xyz), 400_000, replace=False)
        xyz, cls = xyz[idx], cls[idx]
    local = xyz - center
    keep = denoise(local)
    local, cls = local[keep], cls[keep]
    pred = iptd(local)
    # 只在有明确分类的点上比：class 2 = ground, class 1 = unassigned/non-ground in this file
    labeled = (cls == 1) | (cls == 2)
    if labeled.sum() < 1000:
        print("[iptd] SKIP: not enough labeled points")
        return
    truth = cls[labeled] == 2
    agree = (pred[labeled] == truth).mean()
    ok = agree >= 0.90
    print(f"[iptd] agreement with class-2/1 labels: {agree:.1%} -> {'PASS' if ok else 'FAIL'} (>=90%)")


def run_full(out_dir):
    from deform.pipeline import run

    results = run(str(BASE), str(TARGET), str(ROI), str(out_dir), buffer=50.0)
    r = results[0]
    print(json.dumps({k: r[k] for k in ("target_id", "coverage", "counts", "lod_median",
                                         "registration_error", "vmax", "registration", "warnings", "timings")},
                     indent=2, ensure_ascii=False))
    # 定性：样例 RoI 内应有米级正形变（填方）
    mp = r.get("max_positive")
    if mp and mp["value"] > 1.0:
        print(f"[full] qualitative uplift ok: max_positive={mp['value']:.2f} m")
    else:
        print(f"[full] qualitative uplift weak/missing: {mp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "output" / "validate"))
    args = ap.parse_args()
    roi = json.loads(ROI.read_text(encoding="utf-8"))
    center = np.asarray(roi["center"], float)
    w, h = float(roi["w"]), float(roi["h"])
    check_denoise_safety(center, w, h)
    check_iptd_agreement(center, w, h)
    if args.full:
        run_full(args.out)
    else:
        print("skip full pipeline (pass --full to run)")


if __name__ == "__main__":
    main()
