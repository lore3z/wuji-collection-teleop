"""Repository-local launch paths for the checksum-verified teleop sources."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
LINKER_PACKAGE = ROOT / 'vendor/linkerhand_retarget'


def _link(target, link):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        raise RuntimeError(f'Portable runtime path is occupied: {link}')
    link.symlink_to(target, target_is_directory=target.is_dir())


def portable_home():
    # Only launch-path expressions in the frozen module use this directory.
    # The real user HOME and SDK calibration profile are unaffected.
    home = ROOT / '.runtime/portable-home'
    workspace = home / 'linkerhand-telop-ros2'
    workspace.mkdir(parents=True, exist_ok=True)
    _link(LINKER_PACKAGE, workspace / 'src/linkerhand_retarget/linkerhand_retarget')
    _link(ROOT / 'runtime/l20_wuji_hw_bridge.py', workspace / 'tools/l20_wuji_hw_bridge.py')
    _link(ROOT, home / 'wuji_ftp1_collection_v8 (1)')
    _link(ROOT, home / 'wuji_things/wuji-retargeting')
    setup = workspace / 'install/setup.bash'
    setup.parent.mkdir(parents=True, exist_ok=True)
    if not setup.exists():
        # This legacy launcher needs ROS messages. The official CAN driver
        # runs separately through l20.sh and its own project-local overlay.
        setup.write_text('source /opt/ros/humble/setup.bash\n')
    return home


def human_urdf():
    configured = os.environ.get('WUJI_HUMAN_URDF')
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f'WUJI_HUMAN_URDF does not exist: {path}')
        return path
    candidates = sorted((Path.home() / '.wuji/sdk/users').glob('*/models/right_hand.urdf'))
    if len(candidates) != 1:
        raise RuntimeError('Set WUJI_HUMAN_URDF to the current Wuji SDK user right_hand.urdf')
    return candidates[0]
