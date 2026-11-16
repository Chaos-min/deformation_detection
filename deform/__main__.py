"""命令行入口：python -m deform --base ... --target ... --roi ... --out ..."""
import argparse
import json
import sys
from pathlib import Path

from deform.pipeline import DeformError, run


def main(argv=None):
    p = argparse.ArgumentParser(description="面-两期山体形变位移算法")
    p.add_argument("--base", required=True, help="基准期 LAS")
    p.add_argument("--target", required=True, help="目标期 LAS")
    p.add_argument("--roi", required=True, help="mountain_roi.json")
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--buffer", type=float, default=50.0, help="RoI 外扩缓冲带（米）")
    p.add_argument("--vmax", type=float, default=None, help="热图色标半幅（米），默认自动")
    p.add_argument("--params", default=None, help="参数覆盖 JSON")
    args = p.parse_args(argv)

    params = None
    if args.params:
        params = json.loads(Path(args.params).read_text(encoding="utf-8"))
    try:
        results = run(
            args.base,
            args.target,
            args.roi,
            args.out,
            buffer=args.buffer,
            vmax=args.vmax,
            params=params,
        )
    except DeformError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for r in results:
        print(f"{r['target_id']}: significant={r['counts']['significant']} "
              f"vmax={r['vmax']:.3f} reg={r['registration']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
