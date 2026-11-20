"""地面点稳定区配准：优先 py4dgeo.icp_with_stable_areas，失败则单位变换或备用环带 ICP。"""
import numpy as np
import py4dgeo
from scipy.spatial import cKDTree


def voxel_downsample(xyz, voxel=0.25):
    """每个 voxel×voxel 平面格保留第一个点。"""
    if len(xyz) == 0:
        return xyz
    ij = np.floor((xyz[:, :2] - xyz[:, :2].min(0)) / voxel).astype(np.int64)
    _, idx = np.unique(ij, axis=0, return_index=True)
    return xyz[np.sort(idx)]


def _mad(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.inf
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def _apply(xyz, matrix, reduction_point):
    """Tx = R(x-x0) + t + x0，matrix 为 4×4。"""
    R, t = matrix[:3, :3], matrix[:3, 3]
    x0 = np.asarray(reduction_point, float)
    return (xyz - x0) @ R.T + t + x0


def _invert(xyz, matrix, reduction_point):
    """_apply 的逆。"""
    R, t = matrix[:3, :3], matrix[:3, 3]
    x0 = np.asarray(reduction_point, float)
    return (xyz - t - x0) @ R + x0


def _rotation_deg(matrix):
    R = matrix[:3, :3]
    # 稳健：由迹求旋转角
    c = (np.trace(R) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _m3c2_gaps(ref, tgt):
    """沿局部法线的 M3C2 距离，供配准残差诊断。配准本身仍用原来的稳定区 ICP。"""
    core = voxel_downsample(tgt, 1.0)
    if len(core) < 30 or len(ref) < 30:
        return np.array([])
    algo = py4dgeo.M3C2(
        epochs=(
            py4dgeo.Epoch(np.ascontiguousarray(tgt, np.float64)),
            py4dgeo.Epoch(np.ascontiguousarray(ref, np.float64)),
        ),
        corepoints=np.ascontiguousarray(core, np.float64),
        normal_radii=[0.5, 1.0, 2.0],
        cyl_radius=0.5,
        max_distance=10.0,
        registration_error=0.0,
        orientation_vector=np.array([0.0, 0.0, 1.0]),
    )
    distances, _ = algo.run()
    return -np.asarray(distances, float)


def _point_to_plane_residuals(ref, ref_normals, tgt):
    _, idx = cKDTree(ref).query(tgt, workers=-1)
    d = np.einsum("ni,ni->n", tgt - ref[idx], ref_normals[idx])
    return d


def _identity_info(status, warnings, residual_before=None, residual_after=None):
    I = np.eye(4)
    return I, {
        "status": status,
        "matrix": I.tolist(),
        "translation": [0.0, 0.0, 0.0],
        "rotation_deg": 0.0,
        "residual_before": residual_before,
        "residual_after": residual_after if residual_after is not None else residual_before,
        "warnings": warnings,
    }


def _fallback_ring_icp(ref_ground, tgt_ground, reduction_point, ring_mask_tgt, voxel,
                       max_iterations=20, mad_factor=3.0):
    """RoI 外缓冲环上的点到面 ICP，迭代剔除大残差点。"""
    ref = voxel_downsample(ref_ground, voxel)
    tgt0 = voxel_downsample(tgt_ground[ring_mask_tgt] if ring_mask_tgt is not None else tgt_ground, voxel)
    if len(ref) < 100 or len(tgt0) < 100:
        return None, "fallback: too few ring points"
    ref_e = py4dgeo.Epoch(np.ascontiguousarray(ref, np.float64))
    ref_e.calculate_normals(radius=1.0)
    normals = np.asarray(ref_e.normals)
    keep = np.ones(len(tgt0), bool)
    matrix = np.eye(4)
    for _ in range(max_iterations):
        tgt = _apply(tgt0[keep], matrix, reduction_point)
        epoch = py4dgeo.Epoch(np.ascontiguousarray(tgt, np.float64))
        trafo = py4dgeo.point_to_plane_icp(ref_e, epoch, reduction_point=reduction_point)
        step = np.asarray(trafo.affine_transformation, float)
        # 合成：先旧变换，再新变换（都相对同一 reduction_point）
        # Tx' = R2(Tx - x0) + t2 + x0 = R2 R1 (x-x0) + R2 t1 + t2 + x0
        R1, t1 = matrix[:3, :3], matrix[:3, 3]
        R2, t2 = step[:3, :3], step[:3, 3]
        matrix = np.eye(4)
        matrix[:3, :3] = R2 @ R1
        matrix[:3, 3] = R2 @ t1 + t2
        aligned = _apply(tgt0[keep], matrix, reduction_point)
        res = _point_to_plane_residuals(ref, normals, aligned)
        thr = mad_factor * _mad(res)
        if not np.isfinite(thr) or thr <= 0:
            break
        ok = np.abs(res) <= thr
        if ok.mean() > 0.98 or ok.sum() < 50:
            break
        # 更新 keep：在当前 keep 子集上再筛
        idx = np.flatnonzero(keep)
        keep[:] = False
        keep[idx[ok]] = True
    return matrix, None


def register(
    ref_ground,
    tgt_ground,
    reduction_point=None,
    voxel=0.25,
    initial_distance_threshold=0.5,
    level_of_detection=0.05,
    supervoxel_resolution=2.0,
    min_svp_num=10,
    max_translation=0.5,
    max_rotation_deg=0.2,
    residual_worsen_ratio=1.1,
    use_fallback=False,
    ring_mask_tgt=None,
):
    """配准目标期地面点到基准期。

    返回 (4×4 matrix, info)。status ∈ {applied, rejected, fallback}。
    matrix 把目标期点变换到基准期坐标系（仅用于计算）。
    """
    ref_ground = np.asarray(ref_ground, float)
    tgt_ground = np.asarray(tgt_ground, float)
    if reduction_point is None:
        reduction_point = np.mean(ref_ground, axis=0)
    else:
        reduction_point = np.asarray(reduction_point, float)

    ref = voxel_downsample(ref_ground, voxel)
    tgt = voxel_downsample(tgt_ground, voxel)
    if len(ref) < 100 or len(tgt) < 100:
        return _identity_info("rejected", ["register: too few ground points after downsample"])

    ref_e = py4dgeo.Epoch(np.ascontiguousarray(ref, np.float64))
    tgt_e = py4dgeo.Epoch(np.ascontiguousarray(tgt, np.float64))
    ref_e.calculate_normals(radius=1.0)
    tgt_e.calculate_normals(radius=1.0)  # icp_with_stable_areas 要求两期都有法线
    normals = np.asarray(ref_e.normals)
    # 是否采用变换仍看点到面残差。写进结果的残差改用法线方向的 M3C2，避免陡坡上用三维最近点。
    plane_before = _mad(_point_to_plane_residuals(ref, normals, tgt))
    residual_before = _mad(_m3c2_gaps(ref, tgt))
    if not np.isfinite(residual_before):
        residual_before = plane_before

    try:
        trafo = py4dgeo.icp_with_stable_areas(
            ref_e,
            tgt_e,
            initial_distance_threshold=initial_distance_threshold,
            level_of_detection=level_of_detection,
            reference_supervoxel_resolution=supervoxel_resolution,
            supervoxel_resolution=supervoxel_resolution,
            min_svp_num=min_svp_num,
            reduction_point=reduction_point,
        )
        matrix = np.asarray(trafo.affine_transformation, float)
    except Exception as e:  # ponytail: 稳定区 ICP 在稀疏/退化场景可能抛错
        matrix = None
        err = f"icp_with_stable_areas failed: {e}"
    else:
        err = None

    warnings = []
    status = "applied"
    if matrix is None:
        warnings.append(err)
        if use_fallback:
            matrix, ferr = _fallback_ring_icp(ref_ground, tgt_ground, reduction_point, ring_mask_tgt, voxel)
            if matrix is None:
                warnings.append(ferr or "fallback failed")
                return _identity_info("rejected", warnings, residual_before)
            status = "fallback"
        else:
            return _identity_info("rejected", warnings, residual_before)

    translation = matrix[:3, 3].tolist()
    # 含 reduction 时平移向量是 t；位移量级用对中心点的实际位移衡量更稳
    center_shift = np.linalg.norm(_apply(reduction_point[None], matrix, reduction_point)[0] - reduction_point)
    rot = _rotation_deg(matrix)
    aligned = _apply(tgt, matrix, reduction_point)
    plane_after = _mad(_point_to_plane_residuals(ref, normals, aligned))
    residual_after = _mad(_m3c2_gaps(ref, aligned))
    if not np.isfinite(residual_after):
        residual_after = plane_after

    reject = False
    if center_shift > max_translation or rot > max_rotation_deg:
        warnings.append(f"transform too large: shift={center_shift:.3f}m rot={rot:.3f}deg")
        reject = True
    if plane_after > plane_before * residual_worsen_ratio and np.isfinite(plane_before):
        warnings.append(
            f"residual worsened: before={plane_before:.4f} after={plane_after:.4f}"
        )
        reject = True

    if reject:
        if use_fallback and status != "fallback":
            fb, ferr = _fallback_ring_icp(ref_ground, tgt_ground, reduction_point, ring_mask_tgt, voxel)
            if fb is not None:
                aligned = _apply(tgt, fb, reduction_point)
                residual_after = _mad(_m3c2_gaps(ref, aligned))
                return fb, {
                    "status": "fallback",
                    "matrix": fb.tolist(),
                    "translation": fb[:3, 3].tolist(),
                    "rotation_deg": _rotation_deg(fb),
                    "residual_before": residual_before,
                    "residual_after": residual_after,
                    "warnings": warnings + ([ferr] if ferr else []),
                }
            if ferr:
                warnings.append(ferr)
        return _identity_info("rejected", warnings, residual_before, residual_after)

    return matrix, {
        "status": status,
        "matrix": matrix.tolist(),
        "translation": translation,
        "rotation_deg": rot,
        "residual_before": residual_before,
        "residual_after": residual_after,
        "warnings": warnings,
    }


# 供 pipeline 复用
apply_transform = _apply
invert_transform = _invert
