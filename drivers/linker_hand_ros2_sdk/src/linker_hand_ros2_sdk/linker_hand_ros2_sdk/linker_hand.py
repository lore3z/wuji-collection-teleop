#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
'''
编译: colcon build --symlink-install
启动命令:ros2 run linker_hand_ros2_sdk linker_hand_sdk
'''
from re import A
import copy
import rclpy,sys                                     # ROS2 Python接口库
import time
import numpy as np
from rclpy.node import Node                      # ROS2 节点类
from rclpy.clock import Clock
from std_msgs.msg import String, Header, Float32MultiArray
from sensor_msgs.msg import JointState, PointCloud2, PointField
import time, json, threading
from linker_hand_ros2_sdk.LinkerHand.linker_hand_api import LinkerHandApi
from linker_hand_ros2_sdk.LinkerHand.utils.color_msg import ColorMsg
from linker_hand_ros2_sdk.LinkerHand.utils.open_can import OpenCan


class LinkerHand(Node):
    def __init__(self, name):
        super().__init__(name)
        # 声明参数（带默认值）
        self.declare_parameter('hand_type', 'left')
        self.declare_parameter('hand_joint', 'L6')
        self.declare_parameter('is_touch', False)
        self.declare_parameter('can', 'can0')
        self.declare_parameter('modbus', "None")        

        # ros时间获取
        self.stamp_clock = Clock()
        # 获取参数值
        self.hand_type = self.get_parameter('hand_type').value
        self.hand_joint = self.get_parameter('hand_joint').value
        self.is_touch = self.get_parameter('is_touch').value
        self.can = self.get_parameter('can').value
        self.modbus = self.get_parameter('modbus').value
        self.sdk_v = 2
        self.sleep_time = 0.005
        self.cmd_lock = False
        self.declare_parameter('control_hz', 100.0)
        self.declare_parameter('state_publish_hz', 60.0)
        self.declare_parameter('realtime_g20', False)
        self.declare_parameter('feedback_hz', 60.0)
        # ``staggered`` preserves the vendor scheduling.  ``full_scan`` is a
        # data-collection mode: acquire all five tactile matrices as one
        # coherent scan before publishing its source timestamp.
        self.declare_parameter('tactile_scan_mode', 'staggered')
        self.declare_parameter('tactile_scan_hz', 25.0)
        self.declare_parameter('tactile_reply_wait_ms', 7.0)
        self.declare_parameter('g20_fast_feedback', False)
        self.declare_parameter('feedback_reply_wait_ms', 4.0)
        self.control_hz = float(self.get_parameter('control_hz').value)
        self.state_publish_hz = float(self.get_parameter('state_publish_hz').value)
        self.realtime_g20 = bool(self.get_parameter('realtime_g20').value)
        self.feedback_hz = float(self.get_parameter('feedback_hz').value)
        self.tactile_scan_mode = str(
            self.get_parameter('tactile_scan_mode').value
        ).strip().lower()
        self.tactile_scan_hz = float(
            self.get_parameter('tactile_scan_hz').value
        )
        self.tactile_reply_wait_s = float(
            self.get_parameter('tactile_reply_wait_ms').value
        ) / 1000.0
        self.g20_fast_feedback = bool(
            self.get_parameter('g20_fast_feedback').value
        )
        self.feedback_reply_wait_s = float(
            self.get_parameter('feedback_reply_wait_ms').value
        ) / 1000.0
        if (
            self.control_hz <= 0 or self.state_publish_hz <= 0
            or self.feedback_hz <= 0 or self.tactile_scan_hz <= 0
            or self.tactile_reply_wait_s <= 0
            or self.feedback_reply_wait_s <= 0
        ):
            raise ValueError('all configured frequencies must be positive')
        if self.tactile_scan_mode not in ('staggered', 'full_scan'):
            raise ValueError(
                "tactile_scan_mode must be 'staggered' or 'full_scan'"
            )
        if self.tactile_scan_mode == 'full_scan' and self.hand_joint != 'G20':
            raise ValueError('full_scan is currently implemented only for G20/L20')
        self.last_hand_post_cmd = None # 最新手指位置命令
        self.last_hand_vel_cmd = None # 最新手指速度命令
        self.last_hand_eff_cmd = None # 最新手指力矩命令

        self.last_hand_state = [-1] * 10
        self.last_hand_vel = [-1] * 10
        self.force = [[-1] * 5] * 4
        self.matrix_dic = {
            "stamp":{
                "sec": 0,
                "nanosec": 0,
            },
            "thumb_matrix":[[-1] * 6 for _ in range(12)],
            "index_matrix":[[-1] * 6 for _ in range(12)],
            "middle_matrix":[[-1] * 6 for _ in range(12)],
            "ring_matrix":[[-1] * 6 for _ in range(12)],
            "little_matrix":[[-1] * 6 for _ in range(12)],
            "palm_matrix":[-1],
        }
        self.matrix_lock = threading.Lock()
        self.tactile_scan_seq = 0
        self.next_tactile_scan_monotonic = 0.0
        # 压感矩阵合值，单位g 克
        self.matrix_mass_dic = {
            "stamp":{
                "secs": 0,
                "nsecs": 0,
            },
            "thumb_mass":[-1],
            "index_mass":[-1],
            "middle_mass":[-1],
            "ring_mass":[-1],
            "little_mass":[-1]
        }
        self.last_hand_info = {
            "version": [-1], # Dexterous hand version number
            "hand_joint": self.hand_joint, # Dexterous hand joint type
            "speed": [-1] * 10, # Current speed threshold of the dexterous hand
            "current": [-1] * 10, # Current of the dexterous hand
            "fault": [-1] * 10, # Current fault of the dexterous hand
            "motor_temperature": [-1] * 10, # Current motor temperature of the dexterous hand
            "torque": [-1] * 10, # Current torque of the dexterous hand
            "is_touch":self.is_touch,
            "touch_type": -1,
            "finger_order": None # Finger motor order
        }
        self.version = []
        self.touch_type = -1
        self.hz = 1.0 / self.state_publish_hz

        self.hand_setting_sub = self.create_subscription(String,'/cb_hand_setting_cmd', self.hand_setting_cb, 10)
        self._init_hand()
        time.sleep(1)
        self.run_count = 0 # 计数器，用于记录运行次数
        self.timer = self.create_timer(1.0 / self.control_hz, self.run)
        self.thread_pub_state = threading.Thread(target=self.pub_state)
        self.thread_pub_state.daemon = True
        self.thread_pub_state.start()

    def _init_hand(self):
        self.api = LinkerHandApi(hand_type=self.hand_type, hand_joint=self.hand_joint,modbus=self.modbus,can=self.can)
        time.sleep(0.1)
        self.touch_type = self.api.get_touch_type()
        self.hand_cmd_sub = self.create_subscription(JointState, f'/cb_{self.hand_type}_hand_control_cmd', self.hand_control_cb,10)
        self.hand_state_pub = self.create_publisher(JointState, f'/cb_{self.hand_type}_hand_state',10)
        self.hand_info_pub = self.create_publisher(String, f'/cb_{self.hand_type}_hand_info', 10)
        if self.is_touch == True:
            if self.modbus != "None":
                self.matrix_touch_pub = self.create_publisher(String, f'/cb_{self.hand_type}_hand_matrix_touch', 10)
                #self.matrix_touch_pub_pc = self.create_publisher(PointCloud2, f'/cb_{self.hand_type}_hand_matrix_touch_pc', 10)
                self.matrix_touch_mass_pub = self.create_publisher(String, f'/cb_{self.hand_type}_hand_matrix_touch_mass', 10)
            elif self.touch_type > 1:
                ColorMsg(msg=f"{self.hand_type} {self.hand_joint} Equipped with matrix pressure sensing", color='green')
                self.matrix_touch_pub = self.create_publisher(String, f'/cb_{self.hand_type}_hand_matrix_touch', 10)
                #self.matrix_touch_pub_pc = self.create_publisher(PointCloud2, f'/cb_{self.hand_type}_hand_matrix_touch_pc', 10)
                self.matrix_touch_mass_pub = self.create_publisher(String, f'/cb_{self.hand_type}_hand_matrix_touch_mass', 10)
            elif self.touch_type != -1 and self.modbus == "None":
                ColorMsg(msg=f"{self.hand_type} {self.hand_joint} Equipped with pressure sensor", color="green")
                self.touch_pub = self.create_publisher(Float32MultiArray, f'/cb_{self.hand_type}_hand_force', 10)
            else:
                ColorMsg(msg=f"{self.hand_type} {self.hand_joint} Not equipped with any pressure sensors", color="red")
                self.is_touch = False
            
        self.embedded_version = self.api.get_embedded_version()
        pose = None
        torque = [200, 200, 200, 200, 200]
        speed = [200, 250, 250, 250, 250]
        if self.hand_joint.upper() == "O6" or self.hand_joint.upper() == "L6" or self.hand_joint.upper() == "L6P":
            pose = [200, 255, 255, 255, 255, 180]
            torque = [250, 250, 250, 250, 250, 250]
            # O6 最大速度阈值
            speed = [200, 250, 250, 250, 250, 250]
        elif self.hand_joint == "L7":
            # The data length of L7 is 7, reinitialize here
            pose = [255, 200, 255, 255, 255, 255, 180]
            torque = [250, 250, 250, 250, 250, 250, 250]
            speed = [120, 250, 250, 250, 250, 250, 250]
        elif self.hand_joint == "L10":
            torque = [255] * 10
            pose = [255, 200, 255, 255, 255, 255, 180, 180, 180, 41]
            speed = [200, 250, 250, 250, 250, 250, 250, 250, 250, 250]
        elif self.hand_joint == "L20":
            pose = [255,255,255,255,255,255,10,100,180,240,245,255,255,255,255,255,255,255,255,255]
        elif self.hand_joint == "L21":
            pose = [75, 255, 255, 255, 255, 176, 97, 81, 114, 147, 202, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
        elif self.hand_joint == "L25":
            pose = [75, 255, 255, 255, 255, 176, 97, 81, 114, 147, 202, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
        if pose is not None:
            for i in range(1): 
                self.api.set_speed(speed=speed)
                time.sleep(0.1)
                self.api.set_torque(torque=torque)
                time.sleep(0.1)
                self.api.finger_move(pose=pose)
                time.sleep(0.1)

    def list_check(self,pose):
        if isinstance(pose, list) == False:
            return False
        if len(self.last_hand_post_cmd) != len(pose):
            return False
        threshold = 1 if self.realtime_g20 and self.hand_joint == 'G20' else 3
        return any(abs(last - current) >= threshold for last, current in zip(self.last_hand_post_cmd, pose))

    def hand_control_cb(self, msg):
        if self.last_hand_post_cmd == None or self.list_check(msg.position) == True:
            self.last_hand_post_cmd = msg.position
        if self.last_hand_vel_cmd == None or self.list_check(msg.velocity) == True:
            self.last_hand_vel_cmd = msg.velocity
        if self.last_hand_eff_cmd == None or self.list_check(msg.effort) == True:
            self.last_hand_eff_cmd = msg.effort

    def _matrix_touch_requested(self):
        return (
            self.is_touch is True
            and (self.touch_type > 1 or self.modbus != "None")
            and (
                self.matrix_touch_pub.get_subscription_count() > 0
                or self.matrix_touch_mass_pub.get_subscription_count() > 0
            )
        )

    def _mark_matrix_scan_locked(self):
        """Stamp the completion of a real five-finger source scan.

        The JSON publication clock is intentionally not used here.  Consumers
        can therefore distinguish a new tactile scan from a re-publication of
        the same cached matrix.
        """
        current = self.stamp_clock.now().to_msg()
        self.tactile_scan_seq += 1
        stamp = self.matrix_dic["stamp"]
        # Keep both spellings: existing downstream code has used both forms.
        stamp["sec"] = current.sec
        stamp["nanosec"] = current.nanosec
        stamp["secs"] = current.sec
        stamp["nsecs"] = current.nanosec
        stamp["source_scan_seq"] = self.tactile_scan_seq

    def _scan_all_g20_tactile(self):
        """Acquire five matrices serially, then atomically commit one scan.

        G20 tactile replies are request/response CAN traffic.  The SDK itself
        waits 7 ms for each request, so this is deliberately capped by
        ``tactile_scan_hz`` instead of pretending a 120 Hz cache is fresh.
        """
        thumb = self.api.get_thumb_matrix_touch(
            sleep_time=self.tactile_reply_wait_s
        ).tolist()
        index = self.api.get_index_matrix_touch(
            sleep_time=self.tactile_reply_wait_s
        ).tolist()
        middle = self.api.get_middle_matrix_touch(
            sleep_time=self.tactile_reply_wait_s
        ).tolist()
        ring = self.api.get_ring_matrix_touch(
            sleep_time=self.tactile_reply_wait_s
        ).tolist()
        little = self.api.get_little_matrix_touch(
            sleep_time=self.tactile_reply_wait_s
        ).tolist()
        with self.matrix_lock:
            self.matrix_dic["thumb_matrix"] = thumb
            self.matrix_dic["index_matrix"] = index
            self.matrix_dic["middle_matrix"] = middle
            self.matrix_dic["ring_matrix"] = ring
            self.matrix_dic["little_matrix"] = little
            self._mark_matrix_scan_locked()

    def _run_full_tactile_scan_if_due(self):
        now = time.monotonic()
        if now < self.next_tactile_scan_monotonic:
            return
        started = now
        self._scan_all_g20_tactile()
        finished = time.monotonic()
        period = 1.0 / self.tactile_scan_hz
        # Schedule from the intended source cadence rather than from the end
        # of the previous blocking CAN scan.  If a scan overruns its deadline,
        # leave at least the remaining period before the next one; never emit
        # a catch-up burst that would monopolise the bus.
        next_due = self.next_tactile_scan_monotonic + period
        if next_due <= finished:
            next_due = finished + max(0.0, period - (finished - started))
        self.next_tactile_scan_monotonic = next_due

    def _update_staggered_tactile(self):
        """Vendor-compatible tactile schedule with an honest source stamp."""
        if self.run_count == 3:
            value = self.api.get_thumb_matrix_touch(
                sleep_time=self.tactile_reply_wait_s
            ).tolist()
            with self.matrix_lock:
                self.matrix_dic["thumb_matrix"] = value
        if self.run_count == 4:
            value = self.api.get_index_matrix_touch(
                sleep_time=self.tactile_reply_wait_s
            ).tolist()
            with self.matrix_lock:
                self.matrix_dic["index_matrix"] = value
        if self.run_count == 5:
            value = self.api.get_middle_matrix_touch(
                sleep_time=self.tactile_reply_wait_s
            ).tolist()
            with self.matrix_lock:
                self.matrix_dic["middle_matrix"] = value
        if self.run_count == 6:
            value = self.api.get_ring_matrix_touch(
                sleep_time=self.tactile_reply_wait_s
            ).tolist()
            with self.matrix_lock:
                self.matrix_dic["ring_matrix"] = value
        if self.run_count == 7:
            value = self.api.get_little_matrix_touch(
                sleep_time=self.tactile_reply_wait_s
            ).tolist()
            with self.matrix_lock:
                self.matrix_dic["little_matrix"] = value
                self._mark_matrix_scan_locked()
        if self.run_count == 8 and self.hand_joint == "O6":
            value = self.api.get_palm_matrix_touch(sleep_time=0.006)
            with self.matrix_lock:
                self.matrix_dic["palm_matrix"] = value

    def run(self):
        if self.sdk_v == 1:
            self.sleep_time = 0.009
        feedback_divider = max(1, round(self.control_hz / self.feedback_hz))
        should_poll_feedback = (
            not (self.realtime_g20 and self.hand_joint == 'G20')
            or self.run_count % feedback_divider == 0
        )
        if self.hand_state_pub.get_subscription_count() > 0 and should_poll_feedback:
            if self.g20_fast_feedback and self.hand_joint == 'G20':
                self.last_hand_state = self.api.get_state_fast(
                    reply_wait_s=self.feedback_reply_wait_s
                )
                self.last_hand_vel = self.api.get_joint_speed_fast(
                    reply_wait_s=self.feedback_reply_wait_s
                )
            else:
                # Vendor-compatible serial feedback path.
                self.last_hand_state = self.api.get_state()
                time.sleep(0.003)
                self.last_hand_vel = self.api.get_joint_speed()
                time.sleep(0.002)
        if self.cmd_lock == False:
            if self.last_hand_post_cmd != None:
                self.api.finger_move(
                    pose=self.last_hand_post_cmd,
                    realtime=self.realtime_g20 and self.hand_joint == 'G20',
                )
                self.last_hand_post_cmd = None
            if self.last_hand_vel_cmd != None:
                vel = list(self.last_hand_vel_cmd)
                if all(x == 0 for x in vel):
                    pass
                else:
                    if (str(self.hand_joint).upper() == "O6" or str(self.hand_joint).upper() == "L6" or str(self.hand_joint).upper() == "L6P") and len(vel) == 6:
                        speed = vel
                        self.api.set_joint_speed(speed=speed)
                    elif self.hand_joint == "L7" and len(vel) == 7:
                        speed = vel
                        self.api.set_joint_speed(speed=speed)
                    elif self.hand_joint == "L10" and len(vel) == 10:
                        speed = [vel[0],vel[2],vel[3],vel[4],vel[5]]
                        self.api.set_joint_speed(speed=speed)
                    elif self.hand_joint == "L20" and len(vel) == 20:
                        speed = [vel[10],vel[1],vel[2],vel[3],vel[4]]
                        self.api.set_joint_speed(speed=speed)
                    elif self.hand_joint == "L21" and len(vel) == 25:
                        speed = vel
                        self.api.set_joint_speed(speed=speed)
                    elif self.hand_joint == "L25" and len(vel) == 25:
                        speed = vel
                        self.api.set_joint_speed(speed=speed)
                self.last_hand_vel_cmd = None
            if not (self.realtime_g20 and self.hand_joint == 'G20'):
                time.sleep(0.003)
            if self.run_count == 3 and self.is_touch == True and self.touch_type == 1 and self.modbus == "None" and self.touch_pub.get_subscription_count() > 0:
                """单点式压力传感器"""
                self.force = self.api.get_force()
            if self._matrix_touch_requested():
                """矩阵式压力传感器。"""
                if self.tactile_scan_mode == 'full_scan':
                    self._run_full_tactile_scan_if_due()
                else:
                    self._update_staggered_tactile()
                    time.sleep(0.005)
            if self.run_count == 8 and self.hand_info_pub.get_subscription_count() > 0:
                """手部信息"""
                self.last_hand_info = {
                    "version": self.embedded_version, # Dexterous hand version number
                    "hand_joint": self.hand_joint, # Dexterous hand joint type
                    "speed": self.api.get_speed(), # Current speed threshold of the dexterous hand
                    "current": self.api.get_current(), # Current of the dexterous hand
                    "fault": self.api.get_fault(), # Current fault of the dexterous hand
                    "motor_temperature": self.api.get_temperature(), # Current motor temperature of the dexterous hand
                    "torque": self.api.get_torque(), # Current torque of the dexterous hand
                    "is_touch":self.is_touch,
                    "touch_type": self.touch_type,
                    "finger_order": self.api.get_finger_order() # Finger motor order
                }
                
            if self.run_count == 9:
                self.api.clear_faults() # 自动清除错误编码
                self.run_count = 0
            self.run_count += 1
            if not (self.realtime_g20 and self.hand_joint == 'G20'):
                time.sleep(0.003)


    def pub_state(self):
        while True:
            if self.hand_state_pub.get_subscription_count() > 0:
                msg = self.joint_state_msg(self.last_hand_state, self.last_hand_vel)
                self.hand_state_pub.publish(msg)
            if self.is_touch == True and self.touch_type == 1 and self.modbus == "None" and self.touch_pub.get_subscription_count() > 0:
                msg = Float32MultiArray()
                msg.data = [float(val) for sublist in self.force for val in sublist]
                self.touch_pub.publish(msg)
            if self.is_touch == True and (self.touch_type > 1 or self.modbus != "None") and (self.matrix_touch_pub.get_subscription_count() > 0 or self.matrix_touch_mass_pub.get_subscription_count() > 0):
                with self.matrix_lock:
                    matrix_snapshot = copy.deepcopy(self.matrix_dic)
                # The publisher may run faster than the source scan.  Preserve
                # the source timestamp/sequence in this snapshot so consumers
                # can identify repeated cache frames.
                self.pub_matrix_dic(matrix_snapshot)
                self.pub_matrix_mass(dic=matrix_snapshot)
            if self.hand_info_pub.get_subscription_count() > 0:
                msg = String()
                msg.data = json.dumps(self.last_hand_info)
                self.hand_info_pub.publish(msg)
            time.sleep(self.hz)

    def pub_matrix_mass(self, dic):
        """发布矩阵数据合值 单位g 克 JSON格式"""
        msg = String()
        source_stamp = dic.get("stamp", {})
        self.matrix_mass_dic["stamp"]["secs"] = source_stamp.get("secs", 0)
        self.matrix_mass_dic["stamp"]["nsecs"] = source_stamp.get("nsecs", 0)
        self.matrix_mass_dic["stamp"]["source_scan_seq"] = source_stamp.get(
            "source_scan_seq", 0
        )
        self.matrix_mass_dic["thumb_mass"] = sum(sum(row) for row in dic["thumb_matrix"])
        self.matrix_mass_dic["index_mass"] = sum(sum(row) for row in dic["index_matrix"])
        self.matrix_mass_dic["middle_mass"] = sum(sum(row) for row in dic["middle_matrix"])
        self.matrix_mass_dic["ring_mass"] = sum(sum(row) for row in dic["ring_matrix"])
        self.matrix_mass_dic["little_mass"] = sum(sum(row) for row in dic["little_matrix"])
        if dic["palm_matrix"] == [-1]:
            self.matrix_mass_dic["palm_mass"] = [-1]
        else:
            self.matrix_mass_dic["palm_mass"] = sum(sum(row) for row in dic["palm_matrix"])
        self.matrix_mass_dic["unit"] = "g"
        msg.data = json.dumps(self.matrix_mass_dic)
        self.matrix_touch_mass_pub.publish(msg)

    def pub_matrix_point_cloud(self):
        """发布矩阵数据点云格式"""
        tmp_dic = self.matrix_dic.copy()
        del tmp_dic['stamp']               # 去掉时间戳字段
        all_matrices = list(tmp_dic.values())  # 5 帧，每帧 6×12=72 个数
        # 摊平到一维：360 个 float
        flat_list = [v for frame in all_matrices for v in frame]  # 360
        flat = np.concatenate([np.asarray(np.clip(c, 0, 255), dtype=np.uint8) for c in flat_list])
        fields = [PointField(
            name='val',
            offset=0,
            datatype=PointField.UINT8,
            count=1
        )]
        pc = PointCloud2()
        pc.header.stamp = self.stamp_clock.now().to_msg()
        pc.header.frame_id = ''
        pc.height = 1
        pc.width = flat.size         # 360
        pc.fields = fields
        pc.is_bigendian = False
        pc.point_step = 1            # 1 个 float32
        pc.row_step = pc.point_step * pc.width
        pc.data = flat.tobytes()     # 1440 字节
        self.matrix_touch_pub_pc.publish(pc)

    def pub_matrix_dic(self, dic):
        """发布矩阵数据JSON格式"""
        msg = String()
        msg.data = json.dumps(dic)
        self.matrix_touch_pub.publish(msg)

    def joint_state_msg(self, pose,vel=[]):
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = self.api.get_finger_order()
        joint_state.position = [float(x) for x in pose]
        if len(vel) > 1:
            joint_state.velocity = [float(x) for x in vel]
        else:
            joint_state.velocity = [0.0] * len(pose)
        joint_state.effort = [0.0] * len(pose)
        return joint_state
    
    

    

    def hand_setting_cb(self,msg):
        '''控制命令回调'''
        data = json.loads(msg.data)
        print(f"Received setting command: {data['setting_cmd']}",flush=True)
        try:
            if data["params"]["hand_type"] == "left":
                hand = self.api
                hand_left = True
            elif data["params"]["hand_type"] == "right":
                hand = self.api
                hand_right = True
            else:
                print("Please specify the hand part to be set",flush=True)
                return
            self.cmd_lock = True
            # Set maximum torque
            if data["setting_cmd"] == "set_max_torque_limits": # Set maximum torque
                torque = list(data["params"]["torque"])
                hand.set_torque(torque=torque)
                
            if data["setting_cmd"] == "set_speed": # Set speed
                if isinstance(data["params"]["speed"], list) == True:
                    speed = data["params"]["speed"]
                    hand.set_speed(speed=speed)
                else:
                    ColorMsg(msg=f"Speed parameter error, speed must be a list", color="red")
            if data["setting_cmd"] == "clear_faults": # Clear faults
                if hand_left == True and self.hand_joint == "L10" :
                    ColorMsg(msg=f"L10 left hand cannot clear faults")
                elif hand_right == True and self.hand_joint == "L10" :
                    ColorMsg(msg=f"L10 right hand cannot clear faults")
                else:
                    hand.clear_faults()
            if data["setting_cmd"] == "get_faults": # Get faults
                f = hand.get_fault()
                ColorMsg(msg=f"Get faults: {f}")
            if data["setting_cmd"] == "electric_current": # Get current
                ColorMsg(msg=f"Get current: {hand.get_current()}")
            if data["setting_cmd"] == "set_electric_current": # Set current
                if isinstance(data["params"]["current"], list) == True:
                    hand.set_current(data["params"]["current"])
            if data["setting_cmd"] == "show_fun_table": # Get faults
                f = hand.show_fun_table()
        except:
            print("命令参数错误")
            self.cmd_lock = False
        finally:
            self.cmd_lock = False


    def close_can(self):
        self.api.open_can.close_can(can=self.can)
        sys.exit(0)

        
def main(args=None):
    try:
        rclpy.init(args=args)
        node = LinkerHand("linker_hand_sdk")
        embedded_version = node.embedded_version
        if len(embedded_version) == 3 or node.hand_joint.upper() == "O6" or node.hand_joint.upper() == "L6" or node.hand_joint.upper() == "G20":
            ColorMsg(msg=f"New Matrix Touch For SDK V2", color="green")
            node.sdk_v = 2
        elif len(embedded_version) == 6 and node.hand_joint == "L10":
            ColorMsg(msg=f"New Matrix Touch For SDK V2", color="green")
            node.sdk_v = 2
        elif len(embedded_version) > 4 and ((embedded_version[0]==10 and embedded_version[4]>35) or (embedded_version[0]==7 and embedded_version[4]>50) or (embedded_version[0] == 6)):
            ColorMsg(msg=f"New Matrix Touch For SDK V2", color="green")
            node.sdk_v = 2
        else:
            ColorMsg(msg=f"SDK V1", color="green")
            node.sdk_v = 1
        rclpy.spin(node)         # 主循环，监听 ROS 回调
    except KeyboardInterrupt:
        print("收到 Ctrl+C，准备退出...")
    finally:
        # node.close_can()         # 关闭 CAN 或其他硬件资源
        # node.destroy_node()      # 销毁 ROS 节点
        # rclpy.shutdown()         # 关闭 ROS
        print("程序已退出。")
