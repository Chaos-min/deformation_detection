# 面-两期山体形变

基准期与目标期机载点云，按 RoI 裁剪后：去噪 → IPTD 提地面 → 稳定区配准 → M3C2 沿法线测距。只把超过检测限的变化标为显著。正值向外/向上（暖色），负值向内/向下（冷色）。

## 环境

```bash
conda env create -f environment.yml
conda activate deform
```

## 运行

```bash
python -m deform --base 基准.las --target 目标.las --roi data/mountain_roi.json --out output
```

| 参数 | 作用 |
|---|---|
| `--buffer` | RoI 外扩缓冲带，默认 50 m，供配准用 |
| `--vmax` | 色标半幅（米）；省略则按显著形变自动取 |
| `--ground` | 地面提取：`iptd`（默认）或 `tfmptd`（多尺度 PTD） |
| `--params` | 覆盖 JSON，键为 `denoise` / `iptd` / `tfmptd` / `register` / `change` |
| `--stages` | 额外写出各阶段点云，供算法自查 |

`roi.json` 可以是一个对象或对象列表（多个山体逐个处理）。`w` 为东西向全长，`h` 为南北向全长；`l` 不参与裁剪。

## 输出

每个 `target_id` 一个目录：

| 文件 | 内容 |
|---|---|
| `ground_heatmap.las` | RoI 内地面点，RGB 为形变色 |
| `roi_heatmap.las` | RoI 内全部点；非地面点为灰 |
| `colorbar.png` | 距离与颜色 |
| `result.json` | 最大正/负形变及坐标、检测限、配准状态、点数、耗时 |

两个 LAS 的坐标仍是目标期原始坐标。附加字段：`deformation`（米）、`lod`（米）、`significant`。

`--stages` 时多一个 `stages/`：`01_clip_*`、`02_denoise_*`、`03_ground_*`、`04_registered_target`（已变到基准期坐标系）、`05_corepoints`（带形变字段）。

## 检查

```bash
python -m pytest tests/test_deform.py
python tests/validate_real.py          # 需要 data/ 里的两期 LAS
python tests/validate_real.py --full   # 跑真实全流程
```

裸露平缓面的检测限大约 4–5 cm，植被陡坡更差。RoI 连同缓冲带一起移动时测不出来。
