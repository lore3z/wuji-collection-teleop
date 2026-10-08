#!/usr/bin/env python3
"""Verify that a connected Wuji glove satisfies the official 526-taxel contract.

This is a preflight diagnostic, not a calibration routine.  It deliberately
does not fabricate missing taxels, interpolate a quiet row, or modify firmware.
"""

from __future__ import annotations

# Resolve project imports independently of the current working directory.
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_root = _ProjectPath(__file__).resolve().parents[1]
for _project_path in (_project_root, _project_root / "src"):
    if str(_project_path) not in _project_sys.path:
        _project_sys.path.insert(0, str(_project_path))


import argparse
import json
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wuji_glove_d435_collect import (
    DEFAULT_GLOVE_SN,
    GLOVE_SHAPE,
    WUJI_OFFICIAL_ACTIVE_TAXELS,
    _glove_tactile,
    _glove_tactile_zones,
    tactile_active_taxel_count,
)


# Current official zone counts in the same order as _glove_tactile_zones().
OFFICIAL_ZONE_ACTIVE = {
    "thumb": 41,
    "index": 43,
    "middle": 58,
    "ring": 52,
    "pinky": 44,
    "palm": 288,
}


def _seq(value: object) -> int:
    header = getattr(value, "header", None)
    return int(getattr(header, "sequence", getattr(header, "seq", 0)) or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glove-sn", default=DEFAULT_GLOVE_SN)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--json-report", type=Path, default=None,
                        help="write a concise support report to this JSON file")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    from wuji_sdk import SdkManager

    glove = SdkManager.instance().connect(sn=args.glove_sn, device_name="tactile_contract_check")
    tactile_sub = zones_sub = cloud_sub = None
    try:
        tactile_sub = glove.tactile().subscribe()
        zones_sub = glove.tactile_zones().subscribe()
        cloud_sub = glove.tactile_point_cloud().subscribe()
        matrix_counts: list[int] = []
        zone_counts: list[int] = []
        cloud_counts: list[int] = []
        row10_values: list[float] = []
        row6_values: list[float] = []
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            frame = tactile_sub.recv()
            if frame is not None:
                raw = _glove_tactile(frame)
                if raw is not None:
                    matrix_counts.append(tactile_active_taxel_count(raw))
                    row10_values.extend(raw[10][raw[10] >= 0].tolist())
                    row6_values.extend(raw[6][raw[6] >= 0].tolist())
            frame = zones_sub.recv()
            if frame is not None:
                result = _glove_tactile_zones(frame)
                if result is not None:
                    _stats, counts = result
                    zone_counts.append(int(counts.sum()))
            frame = cloud_sub.recv()
            if frame is not None:
                try:
                    cloud_counts.append(int(frame.point_count()))
                except Exception:
                    try:
                        cloud_counts.append(len(frame.data) // int(frame.point_stride))
                    except Exception:
                        pass
            time.sleep(0.0005)
        latest_zone_counts: dict[str, int] = {}
        # Keep the per-zone evidence. The legacy device currently reports
        # 45/44/58/52/45/290; the official layout is 41/43/58/52/44/288.
        end = time.monotonic() + 0.25
        while time.monotonic() < end and not latest_zone_counts:
            frame = zones_sub.recv()
            if frame is not None:
                try:
                    for name in OFFICIAL_ZONE_ACTIVE:
                        values = np.asarray(getattr(frame, name), dtype=np.float32).reshape(-1)
                        latest_zone_counts[name] = int(np.count_nonzero(np.isfinite(values) & (values >= 0.0)))
                except Exception:
                    latest_zone_counts = {}
            time.sleep(0.0005)
        try:
            sdk_version = version("wuji-sdk")
        except PackageNotFoundError:
            sdk_version = "unknown"

        def summary(label: str, values: list[int]) -> str:
            return f"{label}: count={len(values)}, unique={sorted(set(values)) or 'none'}"
        print("Wuji tactile contract preflight")
        print("  wuji-sdk version:", sdk_version)
        print("  expected active taxels:", WUJI_OFFICIAL_ACTIVE_TAXELS, "of", int(np.prod(GLOVE_SHAPE)))
        print(" ", summary("full 24x31 raw", matrix_counts))
        print(" ", summary("vendor tactile_zones total", zone_counts))
        print(" ", summary("tactile point cloud", cloud_counts))
        if latest_zone_counts:
            expected = ", ".join(f"{name}={OFFICIAL_ZONE_ACTIVE[name]}" for name in OFFICIAL_ZONE_ACTIVE)
            actual = ", ".join(f"{name}={latest_zone_counts[name]}" for name in OFFICIAL_ZONE_ACTIVE)
            print("  official zone active:", expected)
            print("  device zone active  :", actual)
        for row, values in ((10, row10_values), (6, row6_values)):
            nonzero = int(np.count_nonzero(np.asarray(values) > 0.05))
            print(f"  row {row} diagnostic: valid samples={len(values)}, >0.05 samples={nonzero}, max={max(values, default=float('nan')):.4f}")
        good = (
            bool(matrix_counts)
            and bool(zone_counts)
            and bool(cloud_counts)
            and set(matrix_counts) == {WUJI_OFFICIAL_ACTIVE_TAXELS}
            and set(zone_counts) == {WUJI_OFFICIAL_ACTIVE_TAXELS}
            and set(cloud_counts) == {WUJI_OFFICIAL_ACTIVE_TAXELS}
        )
        report = {
            "glove_sn": args.glove_sn,
            "wuji_sdk_version": sdk_version,
            "expected_active_taxels": WUJI_OFFICIAL_ACTIVE_TAXELS,
            "raw_active_counts": sorted(set(matrix_counts)),
            "zone_total_counts": sorted(set(zone_counts)),
            "point_cloud_counts": sorted(set(cloud_counts)),
            "official_zone_active": OFFICIAL_ZONE_ACTIVE,
            "device_zone_active": latest_zone_counts,
            "passed": good,
        }
        if args.json_report is not None:
            path = args.json_report.expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"  report: {path}")
        if not good:
            print("[FAIL] Firmware/SDK tactile mapping is inconsistent. Do not collect FTP-1 training episodes.")
            print("       Contact-model calibration only changes tactile_residual/tactile_binary, never raw/zones masks.")
            print("       Update the Wuji glove firmware using the official matching SDK/firmware procedure, reconnect, then rerun this check.")
            if latest_zone_counts:
                surplus = {name: latest_zone_counts[name] - OFFICIAL_ZONE_ACTIVE[name] for name in OFFICIAL_ZONE_ACTIVE}
                print("       zone surplus vs official:", surplus)
            raise SystemExit(2)
        print("[PASS] All three source representations use the official 526-taxel contract.")
    finally:
        for sub in (tactile_sub, zones_sub, cloud_sub):
            try:
                if sub is not None:
                    sub.close()
            except Exception:
                pass
        try:
            glove.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        # Connection loss is an operator-facing preflight failure, not a
        # Python programming error. Keep the SDK details in ~/.wuji/logs and
        # leave the collection console concise so the launcher can retry.
        print(f"[FAIL] Wuji 连接/数据流检查失败: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
