#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('control_hz', default_value='100.0'),
        DeclareLaunchArgument('state_publish_hz', default_value='60.0'),
        DeclareLaunchArgument('feedback_hz', default_value='60.0'),
        DeclareLaunchArgument('realtime_g20', default_value='false'),
        DeclareLaunchArgument('tactile_scan_mode', default_value='staggered'),
        DeclareLaunchArgument('tactile_scan_hz', default_value='25.0'),
        DeclareLaunchArgument('tactile_reply_wait_ms', default_value='7.0'),
        DeclareLaunchArgument('g20_fast_feedback', default_value='false'),
        DeclareLaunchArgument('feedback_reply_wait_ms', default_value='4.0'),
        Node(
            package='linker_hand_ros2_sdk',
            executable='linker_hand_sdk',
            name='linker_hand_sdk',
            output='screen',
            parameters=[{
                'hand_type': 'right', # 配置Linker Hand灵巧手类型 left | right 字母为小写
                'hand_joint': "G20", # O6\L6P\L6\L7\L10\L20\G20(工业版)\L21 字母为大写
                'is_touch': True, # 配置Linker Hand灵巧手是否有压力传感器 True | False
                'can': 'can0', # 这里需要修改为实际的CAN总线名称 如果是win系统则类似于 PCAN_USBBUS1。注：蓝色盒子为Linux下can0，WIN下位PCAN_USBBUS1。透明盒子Linux下为can0，WIN下为0
                "modbus": "None", # "None" | "/dev/ttyUSB0" 这里需要修改为实际的Modbus总线名称 如果是win系统则 COM* Ubuntu则为/dev/ttyUSB* 注意添加sudo chmod 777 /dev/ttyUSB*权限
                # Defaults keep the vendor behaviour.  The V7 teleoperation
                # profile supplies 120 / 120 / 60 / true explicitly.
                'control_hz': LaunchConfiguration('control_hz'),
                'state_publish_hz': LaunchConfiguration('state_publish_hz'),
                'feedback_hz': LaunchConfiguration('feedback_hz'),
                'realtime_g20': LaunchConfiguration('realtime_g20'),
                'tactile_scan_mode': LaunchConfiguration('tactile_scan_mode'),
                'tactile_scan_hz': LaunchConfiguration('tactile_scan_hz'),
                'tactile_reply_wait_ms': LaunchConfiguration('tactile_reply_wait_ms'),
                'g20_fast_feedback': LaunchConfiguration('g20_fast_feedback'),
                'feedback_reply_wait_ms': LaunchConfiguration('feedback_reply_wait_ms'),
            }],
        ),
    ])
