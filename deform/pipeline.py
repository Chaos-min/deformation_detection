"""流程编排：RoI 读取、LAS 裁剪、去噪→IPTD→配准→M3C2→写出。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import laspy
import numpy as np

from deform.change import (
    choose_vmax,
    colorize,
    extrema,
    m3c2,
    propagate_to_points,
    voxel_corepoints,
    write_colorbar,
)
from deform.denoise import denoise
from deform.iptd import iptd
from deform.tfmptd import tfmptd
from deform.csf_ground import csf_ground
from deform.register import apply_transform, register


class DeformError(Exception):
    """可预期的输入/覆盖率错误，由 CLI 转成非零退出。"""


def load_rois(roi_json):
    data = json.loads(Path(roi_json).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise DeformError("roi.json must be an object or a list of objects")


def _crs_wkt(header):
    try:
        crs = header.parse_crs()
        return crs.to_wkt() if crs is not None else None
    except Exception:
        return None


def read_header_meta(las_path):
    with laspy.open(las_path) as reader:
        h = reader.header
        return {
            "scales": np.array(h.scales, float),
            "offsets": np.array(h.offsets, float),
            "point_format_id": int(h.point_format.id),
            "version": f"{h.version.major}.{h.version.minor}",
            "crs_wkt": _crs_wkt(h),
            "vlrs": list(h.vlrs),
        }


def read_clipped_xyz(las_path, center, w, h, buffer, chunk_size=4_000_000):
    """分块读取，返回 RoI 外扩 buffer 水平范围内的 xyz。"""
    center = np.asarray(center, float)
    half = np.array([w / 2 + buffer, h / 2 + buffer])
    lo, hi = center[:2] - half, center[:2] + half
    parts = []
    with laspy.open(las_path) as reader:
        for pts in reader.chunk_iterator(chunk_size):
            x, y, z = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)
            m = (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1])
            if m.any():
                parts.append(np.column_stack([x[m], y[m], z[m]]))
    if not parts:
        return np.zeros((0, 3), float)
    return np.vstack(parts)


def _in_roi_mask(xy, center_xy, w, h):
    return (np.abs(xy[:, 0] - center_xy[0]) <= w / 2) & (np.abs(xy[:, 1] - center_xy[1]) <= h / 2)


def _coverage(xy, center, w, h, cell=1.0):
    m = _in_roi_mask(xy, center[:2], w, h)
    if not m.any():
        return 0.0
    origin = np.array([center[0] - w / 2, center[1] - h / 2])
    nx, ny = max(1, int(np.ceil(w / cell))), max(1, int(np.ceil(h / cell)))
    ij = np.floor((xy[m, :2] - origin) / cell).astype(np.int64)
    ij = ij[(ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)]
    if len(ij) == 0:
        return 0.0
    return np.unique(ij, axis=0).shape[0] / float(nx * ny)


def write_heatmap_las(path, xyz, rgb, classification, deformation, lod, significant, header_meta):
    """写出带 deformation/lod/significant 附加字段的点云。"""
    n = len(xyz)
    header = laspy.LasHeader(point_format=header_meta["point_format_id"], version=str(header_meta["version"]))
    header.scales = header_meta["scales"]
    header.offsets = header_meta["offsets"]
    header.add_extra_dim(laspy.ExtraBytesParams(name="deformation", type=np.float32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="lod", type=np.float32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="significant", type=np.uint8))
    # 坐标系 VLR：尽量拷贝
    for vlr in header_meta.get("vlrs") or []:
        try:
            header.vlrs.append(vlr)
        except Exception:
            pass
    out = laspy.LasData(header)
    out.x, out.y, out.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out.classification = np.asarray(classification, np.uint8)
    out.red = (np.asarray(rgb[:, 0], np.uint16) * 257)
    out.green = (np.asarray(rgb[:, 1], np.uint16) * 257)
    out.blue = (np.asarray(rgb[:, 2], np.uint16) * 257)
    out.deformation = np.asarray(deformation, np.float32)
    out.lod = np.asarray(np.where(np.isfinite(lod), lod, -1.0), np.float32)
    out.significant = np.asarray(significant, np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.write(str(path))


def write_xyz_las(path, xyz, header_meta, classification=None):
    """只写坐标（和可选分类），供各阶段自查。坐标系与输入一致。"""
    header = laspy.LasHeader(point_format=0, version=str(header_meta["version"]))
    header.scales = header_meta["scales"]
    header.offsets = header_meta["offsets"]
    for vlr in header_meta.get("vlrs") or []:
        try:
            header.vlrs.append(vlr)
        except Exception:
            pass
    out = laspy.LasData(header)
    out.x, out.y, out.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if classification is not None:
        out.classification = np.asarray(classification, np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.write(str(path))


def _merge_params(defaults, overrides):
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in defaults.items()}
    if not overrides:
        return out
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


DEFAULT_PARAMS = {"denoise": {}, "iptd": {}, "tfmptd": {}, "csf": {}, "register": {}, "change": {}}
_GROUND = {"iptd": iptd, "tfmptd": tfmptd, "csf": csf_ground}


def process_roi(base_las, target_las, roi, out_dir, buffer=50.0, vmax=None, params=None, save_stages=False, ground="iptd"):
    params = _merge_params(DEFAULT_PARAMS, params)
    timings, warnings = {}, []
    center = np.asarray(roi["center"], float)
    w, h = float(roi["w"]), float(roi["h"])
    target_id = str(roi.get("target_id", "roi"))
    out_roi = Path(out_dir) / target_id
    out_roi.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    base_meta = read_header_meta(base_las)
    tgt_meta = read_header_meta(target_las)
    base_xyz = read_clipped_xyz(base_las, center, w, h, buffer)
    tgt_xyz = read_clipped_xyz(target_las, center, w, h, buffer)
    timings["read"] = time.perf_counter() - t0

    if len(base_xyz) == 0 or len(tgt_xyz) == 0:
        raise DeformError(f"{target_id}: no overlap between RoI+buffer and point clouds")
    if base_meta.get("crs_wkt") and tgt_meta.get("crs_wkt") and base_meta["crs_wkt"] != tgt_meta["crs_wkt"]:
        raise DeformError("base and target CRS differ")

    cov = _coverage(tgt_xyz, center, w, h)
    if cov < 0.5:
        raise DeformError(f"{target_id}: target coverage {cov:.1%} < 50%")
    if cov < 1.0 - 1e-6:
        warnings.append(f"target coverage {cov:.1%} < 100%")

    base_local = base_xyz - center
    tgt_local = tgt_xyz - center

    t0 = time.perf_counter()
    keep_b = denoise(base_local, **params["denoise"])
    keep_t = denoise(tgt_local, **params["denoise"])
    timings["denoise"] = time.perf_counter() - t0
    base_d, tgt_d = base_local[keep_b], tgt_local[keep_t]

    if ground not in _GROUND:
        raise DeformError(f"unknown ground filter: {ground}")
    filt = _GROUND[ground]
    t0 = time.perf_counter()
    g_b = filt(base_d, **params[ground])
    g_t = filt(tgt_d, **params[ground])
    timings[ground] = time.perf_counter() - t0
    ref_g, tgt_g = base_d[g_b], tgt_d[g_t]

    origin2 = np.zeros(2)
    in_b = _in_roi_mask(ref_g[:, :2], origin2, w, h)
    in_t = _in_roi_mask(tgt_g[:, :2], origin2, w, h)
    if in_b.sum() < 1000 or in_t.sum() < 1000:
        raise DeformError(
            f"{target_id}: ground points in RoI too few (base={int(in_b.sum())}, target={int(in_t.sum())})"
        )

    t0 = time.perf_counter()
    matrix, reg_info = register(
        ref_g,
        tgt_g,
        reduction_point=np.zeros(3),
        ring_mask_tgt=~in_t,
        **params["register"],
    )
    timings["register"] = time.perf_counter() - t0
    warnings.extend(reg_info.get("warnings") or [])

    tgt_d_aligned = apply_transform(tgt_d, matrix, np.zeros(3))
    tgt_g_aligned = tgt_d_aligned[g_t]

    change_params = dict(params["change"])
    core_voxel = change_params.pop("core_voxel", 0.25)
    core = voxel_corepoints(tgt_g_aligned[in_t], core_voxel)
    if len(core) == 0:
        raise DeformError(f"{target_id}: no core points")

    t0 = time.perf_counter()
    deformation, lod, significant, reg_err, _ = m3c2(ref_g, tgt_g_aligned, core, **change_params)
    timings["m3c2"] = time.perf_counter() - t0

    tgt_g_roi = tgt_g[in_t]
    tgt_g_roi_aligned = tgt_g_aligned[in_t]
    d_g, l_g, s_g = propagate_to_points(tgt_g_roi_aligned, core, deformation, lod, significant)

    vmax_used = float(vmax) if vmax is not None else choose_vmax(deformation, significant)
    lod_med = float(np.nanmedian(lod[np.isfinite(lod)])) if np.isfinite(lod).any() else float("nan")
    rgb_g = colorize(d_g, s_g, vmax_used, no_data_mask=~np.isfinite(d_g))

    # RoI 内全部去噪点
    tgt_roi_mask = _in_roi_mask(tgt_d[:, :2], origin2, w, h)
    all_xyz = tgt_d[tgt_roi_mask] + center
    ground_in_roi = g_t[tgt_roi_mask]
    d_all = np.full(len(all_xyz), np.nan)
    l_all = np.full(len(all_xyz), np.nan)
    s_all = np.zeros(len(all_xyz), bool)
    rgb_all = np.full((len(all_xyz), 3), 128, np.uint8)
    d_all[ground_in_roi] = d_g
    l_all[ground_in_roi] = l_g
    s_all[ground_in_roi] = s_g
    rgb_all[ground_in_roi] = rgb_g

    ground_xyz = tgt_g_roi + center
    write_heatmap_las(
        out_roi / "ground_heatmap.las",
        ground_xyz,
        rgb_g,
        np.full(len(ground_xyz), 2, np.uint8),
        np.where(np.isfinite(d_g), d_g, 0.0),
        l_g,
        s_g.astype(np.uint8),
        tgt_meta,
    )
    write_heatmap_las(
        out_roi / "roi_heatmap.las",
        all_xyz,
        rgb_all,
        np.where(ground_in_roi, 2, 1).astype(np.uint8),
        np.where(np.isfinite(d_all), d_all, 0.0),
        l_all,
        s_all.astype(np.uint8),
        tgt_meta,
    )
    write_colorbar(out_roi / "colorbar.png", vmax_used, lod_med)

    stage_files = []
    if save_stages:
        stage = out_roi / "stages"
        # 配准后的点在基准期坐标系，加回 RoI 中心后可与原始点云叠合
        dumps = {
            "01_clip_base.las": base_xyz,
            "01_clip_target.las": tgt_xyz,
            "02_denoise_base.las": base_d + center,
            "02_denoise_target.las": tgt_d + center,
            "03_ground_base.las": ref_g + center,
            "03_ground_target.las": tgt_g + center,
            "04_registered_target.las": tgt_d_aligned + center,
        }
        for name, cloud in dumps.items():
            write_xyz_las(stage / name, cloud, tgt_meta)
            stage_files.append(f"stages/{name}")
        write_heatmap_las(
            stage / "05_corepoints.las",
            core + center,
            colorize(deformation, significant, vmax_used),
            np.full(len(core), 2, np.uint8),
            np.where(np.isfinite(deformation), deformation, 0.0),
            lod,
            significant.astype(np.uint8),
            tgt_meta,
        )
        stage_files.append("stages/05_corepoints.las")

    ext = extrema(core + center, deformation, lod, significant)
    result = {
        "target_id": target_id,
        "base_file": str(base_las),
        "target_file": str(target_las),
        "max_positive": ext["max_positive"],
        "max_negative": ext["max_negative"],
        "lod_median": lod_med,
        "registration_error": reg_err,
        "registration": {k: v for k, v in reg_info.items() if k != "warnings"},
        "coverage": cov,
        "counts": {
            "base_clipped": int(len(base_xyz)),
            "target_clipped": int(len(tgt_xyz)),
            "base_denoised": int(len(base_d)),
            "target_denoised": int(len(tgt_d)),
            "base_ground": int(len(ref_g)),
            "target_ground": int(len(tgt_g)),
            "corepoints": int(len(core)),
            "significant": int(significant.sum()),
        },
        "ground_method": ground,
        "vmax": vmax_used,
        "params": params,
        "timings": timings,
        "warnings": list(dict.fromkeys(warnings + list(reg_info.get("warnings") or []))),
        "stages": stage_files,
    }
    (out_roi / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def run(base_las, target_las, roi_json, out_dir, buffer=50.0, vmax=None, params=None, save_stages=False, ground="iptd"):
    return [
        process_roi(
            base_las, target_las, roi, out_dir,
            buffer=buffer, vmax=vmax, params=params, save_stages=save_stages, ground=ground,
        )
        for roi in load_rois(roi_json)
    ]
