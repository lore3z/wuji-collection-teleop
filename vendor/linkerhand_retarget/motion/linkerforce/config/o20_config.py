# 手指配置常量
FINGER_CONFIGS = {
    # 含义解释：
    # robot_idx：URDF关节序列

    # 拇指旋转关节，对应URDF thumb_cmc_roll
    'thumb_rotate': {
        'name': '拇指旋转',
        'joints': [1],
        'weights': {
            'v1': [1],
            'v2': [1]
        },
        'robot_idx': 0,
        'type': 'thumb',
        'reverse_motion': {
            'v1': False,
            'v2': False
        },
        'dynamic_weight': None,
        'extended_mapping': {
            'enabled': True,
            'scale_factor': 1.0,
            'extended_exp_factor': 10
        }
    },
    # 拇指侧摆3个关节的加权系数，人手的0/1/2序列，对应URDF的第2关节(下标1)
    'thumb_abduction': {
        'name': '拇指侧摆',
        'joints': [0, 1, 2],
        'weights': {
            'v1': [0, 1, 0],
            'v2': [0, 1, 0]
        },
        'robot_idx': 1,
        'type': 'thumb',
        'reverse_motion': {
            'v1': False,
            'v2': False
        },
        'dynamic_weight': None,
        'extended_mapping': {
            'enabled': True,
            'scale_factor': 1,
            'extended_exp_factor': 1.0
        }
    },
    # 拇指根部弯曲3个关节的加权系数，人手的2/3/4序列，对应URDF的第3关节(下标2)
    'thumb_root_flexion': {
        'name': '拇指根部弯曲',
        'joints': [2, 3, 4],
        'weights': {
            'v1': [1, 0, 0],
            'v2': [1, 0, 0]
        },
        'robot_idx': 2,
        'type': 'thumb',
        'reverse_motion': {
            'v1': False,
            'v2': False
        },
        'dynamic_weight': None,
        # {
        #     'trigger_finger': 'thumb_abduction',
        #     'threshold': 0.3,
        #     'low_weight_config': {
        #         'joints': [2, 3, 4],
        #         'weights': [1, 0, 0],
        #         'reverse_motion': False
        #     },
        #     'high_weight_config': {
        #         'joints': [2, 3, 4],
        #         'weights': [0.3, 0.0, 0.7],
        #         'reverse_motion': False
        #     }
        # },
        'extended_mapping': {
            'enabled': True,
            'scale_factor': 1,
            'extended_exp_factor': 30
        }
    },
    # 拇指末端弯曲3个关节的加权系数，人手的2/3/4序列，对应URDF的第4关节(下标3)
    'thumb_end_flexion': {
        'name': '拇指指尖弯曲',
        'joints': [2, 3, 4],
        'weights': {
            'v1': [0, 0.0, 1],
            'v2': [0, 0.0, 1]
        },
        'robot_idx': 3,
        'type': 'thumb',
        'reverse_motion': {
            'v1': False,
            'v2': False
        },
        'dynamic_weight': None,
        'extended_mapping': {
            'enabled': True,
            'scale_factor': 0.9,
            'extended_exp_factor': 1.5
        }
    },
    # 食指ROLL旋转（侧摆）关节的加权系数，人手的5序列，对应URDF的第4关节(下标3)
    'index_roll': {
        'name': '食指',
        'joints': [5],
        'weights': {
            'v1': [1],
            'v2': [1]
        },
        'robot_idx': 4,
        'type': 'finger',
        'reverse_motion': {
            'v1': False,
            'v2': False
        },
        'dynamic_weight': None,
        'extended_mapping': {
            'enabled': True,
            'scale_factor': 1.0,
            'extended_exp_factor': 1.0,
        }
    },
    # 食指弯曲（根部弯曲）的加权系数，人身的6/7/8序列，对应URDF的第4关节(下标3)
    'index_root_flexion': {
        'name': '食指',
        'joints': [6, 7, 8],
        'weights': {
            'v1': [1, 0.0, 0],
            'v2': [1, 0.0, 0]
        },
        'robot_idx': 5,
        'type': 'finger',
        'reverse_motion': {
            'v1': False,
            'v2': False
        },
        'dynamic_weight': None,
        'extended_mapping': {
            'enabled': True,
            'scale_factor': 1,
            'extended_exp_factor': 5
        }
    },
    # 食指弯曲（末端弯曲）的加权系数，人手的6/7/8序列，对应URDF的第4关节(下标3)
    'index_end_flexion': {
        'name': '食指',
        'joints': [6, 7, 8],
        'weights': {
            'v1': [0, 0.0, 1],
            'v2': [0, 0.0, 1]
        },
        'robot_idx': 6,
        'type': 'finger',
        'reverse_motion': {
            'v1': False,
            'v2': False
        },
        'dynamic_weight': None,
        'extended_mapping': {
            'enabled': True,
            'scale_factor': 1,
            'extended_exp_factor': 5
        }
    },
    # 中指ROLL旋转（侧摆）关节的加权系数，人手的5序列，对应URDF的第4关节(下标3)
    'middle_roll': {
        'name': '中指',
        'joints': [9],
        'weights': {
            'v1': [1],
            'v2': [1]
        },
        'robot_idx': 7,
        'type': 'finger',
        'reverse_motion': {
            'v1': False,
            'v2': False
        },
        'dynamic_weight': None,
        'extended_mapping': {
            'enabled': True,
            'scale_factor': 1.0,
            'extended_exp_factor': 1.0
        }
    },
    # 中指弯曲（根部弯曲）的加权系数，人手的10/11/12序列，对应URDF的第6关节(下标5)
    'middle_root_flexion': {
        'name': '中指',
        'joints': [10, 11, 12],
        'weights': {
            'v1': [1, 0.0, 0],
            'v2': [1, 0.0, 0]
        },
        'robot_idx': 8,
        'type': 'finger',
        'reverse_motion': {
            'v1': False,
            'v2': False
        },
        'dynamic_weight': None,
        'extended_mapping': {
            'enabled': True,
            'scale_factor': 1,
            'extended_exp_factor': 5
        }
    },
    # 中指弯曲（末端弯曲）的加权系数，人手的10/11/12序列，对应URDF的第6关节(下标5)
    'middle_end_flexion': {
        'name': '中指',
        'joints': [10, 11, 12],
        'weights': {
            'v1': [0, 0.0, 1],
            'v2': [0, 0.0, 1]
        },
        'robot_idx': 9,
        'type': 'finger',
        'reverse_motion': {
            'v1': False,
            'v2': False
        },
        'dynamic_weight': None,
        'extended_mapping': {
            'enabled': True,
            'scale_factor': 1,
            'extended_exp_factor': 5
        }
    },
    # 无名指ROLL旋转（侧摆）关节的加权系数，人手的5序列，对应URDF的第4关节(下标3)
    'ring_roll': {
        'name': '无名指',
        'joints': [13],
        'weights': {
            'v1': [1],
            'v2': [1]
        },
        'robot_idx': 10,
        'type': 'finger',
        'reverse_motion': {
            'v1': False,
            'v2': False
        },
        'dynamic_weight': None,
        'extended_mapping': {
            'enabled': True,
            'scale_factor': 1.0,
            'extended_exp_factor': 1.0,
        }
    },
    # 无名指弯曲（根部弯曲）的加权系数，人手的14/15/16序列，对应URDF的第8关节(下标7)
    'ring_root_flexion': {
        'name': '无名指',
        'joints': [14, 15, 16],
        'weights': {
            'v1': [1, 0.0, 0],
            'v2': [1, 0.0, 0]
        },
        'robot_idx': 11,
        'type': 'finger',
        'reverse_motion': {
            'v1': False,
            'v2': False
        },
        'dynamic_weight': None,
        'extended_mapping': {
            'enabled': True,
            'scale_factor': 1,
            'extended_exp_factor': 5
        }
    },
    # 无名指弯曲（末端弯曲）的加权系数，人手的14/15/16序列，对应URDF的第8关节(下标7)
    'ring_end_flexion': {
        'name': '无名指',
        'joints': [14, 15, 16],
        'weights': {
            'v1': [0, 0.0, 1],
            'v2': [0, 0.0, 1]
        },
        'robot_idx': 12,
        'type': 'finger',
        'reverse_motion': {
            'v1': False,
            'v2': False
        },
        'dynamic_weight': None,
        'extended_mapping': {
            'enabled': True,
            'scale_factor': 1,
            'extended_exp_factor': 5
        }
    },
    # 小指ROLL旋转（侧摆）关节的加权系数，人手的5序列，对应URDF的第4关节(下标3)
    'pinky_roll': {
        'name': '小指',
        'joints': [17],
        'weights': {
            'v1': [1],
            'v2': [1]
        },
        'robot_idx': 13,
        'type': 'finger',
        'reverse_motion': {
            'v1': False,
            'v2': False
        },
        'dynamic_weight': None,
        'extended_mapping': {
            'enabled': True,
            'scale_factor': 1.0,
            'extended_exp_factor': 1.0,
        }
    },
    # 小指弯曲（根部弯曲）的加权系数，人手的18/19/20序列，对应URDF的第10关节(下标9)
    'pinky_root_flexion': {
        'name': '小指',
        'joints': [18, 19, 20],
        'weights': {
            'v1': [1, 0.0, 0],
            'v2': [1, 0.0, 0]
        },
        'robot_idx': 14,
        'type': 'finger',
        'reverse_motion': {
            'v1': False,
            'v2': False
        },
        'dynamic_weight': None,
        'extended_mapping': {
            'enabled': True,
            'scale_factor': 1,
            'extended_exp_factor': 5
        }
    },
    # 小指弯曲（末端弯曲）的加权系数，人手的18/19/20序列，对应URDF的第10关节(下标9)
    'pinky_end_flexion': {
        'name': '小指',
        'joints': [18, 19, 20],
        'weights': {
            'v1': [0, 0.0, 1],
            'v2': [0, 0.0, 1]
        },
        'robot_idx': 15,
        'type': 'finger',
        'reverse_motion': {
            'v1': False,
            'v2': False
        },
        'dynamic_weight': None,
        'extended_mapping': {
            'enabled': True,
            'scale_factor': 1,
            'extended_exp_factor': 30
        }
    }
}

# 映射顺序
MAPPING_ORDER = [
    'thumb_rotate', 'thumb_abduction', 'thumb_root_flexion', 'thumb_end_flexion',
    'index_roll', 'index_root_flexion', 'index_end_flexion',
    'middle_roll', 'middle_root_flexion', 'middle_end_flexion',
    'ring_roll', 'ring_root_flexion','ring_end_flexion',
    'pinky_roll', 'pinky_root_flexion', 'pinky_end_flexion'
]

# 三态默认配置
MULTI_SEGMENT_CONFIG = {
    'states': [
        'original',
        'opose',
        # 'fist'  # 取消注释启用三段映射
    ],
    'state_names': {
        'original': '张手',
        'opose': 'O手势',
        # 'fist': '握拳'
    }
}
MULTI_SEGMENT_CONFIG_FROZEN = tuple(MULTI_SEGMENT_CONFIG['states'])

ROBOT_ORIGINAL_LEFT = [
    0, -1.197, 0.2, 0.0,
    0.137, 0.0, 0.0,
   -0.002, 0.0, 0.0,
   -0.130, 0.0, 0.0,
   -0.210, 0.0, 0.0
]

ROBOT_ORIGINAL_RIGHT = [
    0, 1.197, 0.2, 0.0,
   -0.137, 0.0, 0.0,
   -0.002, 0.0, 0.0,
   -0.130, 0.0, 0.0,
   -0.210, 0.0, 0.0
]

ROBOT_OPOSE_LEFT = [
   -0.153,-0.561, 0.325, 1.199,
    0.131, 0.761, 0.929,
    0.000, 0.761, 0.929,
   -0.117, 0.761, 0.929,
   -0.210, 0.761, 0.929,
]

ROBOT_OPOSE_RIGHT = [
    0.153, 0.561, 0.325, 1.199,
   -0.131, 0.761, 0.929,
    0.000, 0.761, 0.929,
   -0.117, 0.761, 0.929,
   -0.210, 0.761, 0.929
]

ROBOT_FIST_RIGHT = [
    0.52,  0.0,    0.484,  1.279,
    0.0,   1.645,  1.85,
    0.0,   1.67,   1.709,
    0.0,   1.64,   1.773,
    0.0,   1.724,  1.850
]

ROBOT_FIST_LEFT = [
   -0.52,  0,      0.484,  1.279,
    0.00,  1.645,  1.85,
    0.00,  1.67,   1.709,
    0.00,  1.64,   1.773,
    0.00,  1.724,  1.850
]

# 电机输出约束配置 (20电机)
MOTOR_CONSTRAINTS = {
    'left': [
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
    ],
    'right': [
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
        {'min': 0, 'max': 255, 'enabled': False},
    ]
}
