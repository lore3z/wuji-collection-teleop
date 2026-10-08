from glob import glob
import os
from setuptools import find_packages, setup

package_name = "realman_rm75_output"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools", "numpy", "pin>=3.0"],
    zip_safe=True,
    maintainer="Wuji Tech",
    maintainer_email="support@wuji.tech",
    description="Safe dry-run PICO relative-pose mapper for RealMan RM75",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "realman_rm75_output_node = realman_rm75_output.realman_rm75_output_node:main",
            "rm75_single_axis_test_node = realman_rm75_output.rm75_single_axis_test_node:main",
            "rm75_readonly_preflight = realman_rm75_output.rm75_readonly_preflight:main",
            "rm75_fixed_1mm_test = realman_rm75_output.rm75_fixed_1mm_test:main",
            "rm75_single_axis_teleop = realman_rm75_output.rm75_single_axis_teleop:main",
            "rm75_6dof_teleop = realman_rm75_output.rm75_6dof_teleop:main",
            "rm75_mapping_diagnostics = realman_rm75_output.rm75_mapping_diagnostics:main",
            "rm75_dual_tracker_limited_teleop = realman_rm75_output.rm75_dual_tracker_limited_teleop:main",
            "rm75_joint_sender_fake_test = realman_rm75_output.rm75_joint_sender_fake_test:main",
            "rm75_joint_sender_follow_trial = realman_rm75_output.rm75_joint_sender_fake_test:real_main",
            "rm75_kinematic_shadow = realman_rm75_output.rm75_joint_sender_fake_test:shadow_main",
            "rm75_six_axis_monitor = realman_rm75_output.six_axis_shadow_monitor:main",
            "rm75_offline_trajectory = realman_rm75_output.offline_trajectory:main",
            "rm75_external_pose_follower = realman_rm75_output.external_pose_follower:main",
            "rm75_upper_arm_shadow_observer = realman_rm75_output.upper_arm_shadow_observer:main",
            "rm75_shadow_joint_bridge_fake = realman_rm75_output.shadow_joint_real_bridge:fake_main",
            "rm75_shadow_joint_bridge_real = realman_rm75_output.shadow_joint_real_bridge:real_main",
            "rm75_safe_home_mover = realman_rm75_output.rm75_safe_home_mover:main",
        ],
    },
)
