#!/usr/bin/env python3
"""Launch the audited bridge with repository-local vendor imports and assets."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor'))
sys.path.insert(0, str(ROOT / 'vendor/linkerhand_retarget'))
from runtime.project_paths import portable_home

source_path = ROOT / 'runtime/emg_teleop_baseline_v1/bridge/l20_wuji_hw_bridge.py'
spec = importlib.util.spec_from_file_location('wuji_verified_l20_bridge', source_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
source = source_path.read_text().replace('Path.home()', f'Path({str(portable_home())!r})')
exec(compile(source, str(source_path), 'exec'), module.__dict__)
if __name__ == '__main__':
    module.main()
