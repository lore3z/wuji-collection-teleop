from datetime import datetime
import socket
import time
from threading import Thread
import json
from dataclasses import dataclass
from typing import List, Dict, Union, Any, Optional, Callable
import threading
import numpy as np

NODES_HAND = 24
NO_DATA_TIMEOUT = 1.0  # 无数据超时时间（秒）

@dataclass
class Bone:
    Name: str
    Parent: int
    Location: List[float]
    Rotation: List[float]
    Scale: List[float]

@dataclass
class Parameter:
    Name: str
    Value: Union[float, int, bool]

@dataclass
class DeviceData:
    Bones: List[Bone]
    Parameter: List[Parameter]

class MotionData:
    def __init__(self, raw_data: Dict[str, Any]):
        self.devices = {}
        for device_id, device_content in raw_data.items():
            bones = [Bone(**bone) for bone in device_content["Bones"]]
            parameters = [Parameter(**param) for param in device_content["Parameter"]]
            self.devices[device_id] = DeviceData(Bones=bones, Parameter=parameters)

    def get_device(self, device_id: str) -> DeviceData:
        return self.devices.get(device_id)

    def list_sequence_params(self, device_id: str, prefix: str) -> Dict[str, Union[float, int, bool]]:
        device = self.get_device(device_id)
        if not device:
            raise ValueError(f"Device {device_id} not found")

        return {
            param.Name: param.Value
            for param in device.Parameter
            if param.Name.startswith(prefix) and param.Name[len(prefix):].isdigit()
        }


class UdexRealData:
    def __init__(self):
        self.is_update = False
        self.frame_index = 0
        self.frequency = 0
        self.ns_result = 0
        self.jointangle_rHand = [0.0] * NODES_HAND
        self.jointangle_lHand = [0.0] * NODES_HAND
        # self.jointderict_rHand = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        # self.jointderict_lHand = [-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,1,1]
        self.jointderict_rHand = [1] * NODES_HAND
        self.jointderict_lHand = [1] * NODES_HAND
        self.last_data_time = 0.0
        self.is_data_timeout = False


@dataclass
class TimeoutStatus:
    """超时状态数据结构"""
    is_timeout: bool  # 当前是否超时
    last_receive_time: float  # 上次收到数据的时间戳
    time_since_last_data: float  # 距离上次收到数据的秒数
    frame_index: int  # 当前帧数
    timeout_threshold: float  # 超时阈值
    consecutive_timeout_checks: int  # 连续超时检查次数


class UdexRealScoketUdp:
    def __init__(self, host='0.0.0.0', port=7000, buffer_size=2048, device_id='eric'):
        self.socket_udp = None
        self.udp_thread = None
        self.udp_running = False
        self.isconnect = False
        self.host = host
        self.port = port
        self.device_id = device_id
        self.buffer_size = buffer_size
        self.realmocapdata = UdexRealData()
        self.data_lock = threading.Lock()
        
        # 超时检测相关
        self.no_data_timeout = NO_DATA_TIMEOUT
        self.last_receive_time = 0.0
        self.consecutive_timeout_checks = 0  # 连续超时检查次数
        
        # 回调函数（可选）
        self.on_timeout_callback: Optional[Callable[[TimeoutStatus], None]] = None
        self.on_data_recovered_callback: Optional[Callable[[], None]] = None

    def set_timeout_callback(self, callback: Callable[[TimeoutStatus], None]):
        """设置超时回调函数"""
        self.on_timeout_callback = callback

    def set_data_recovered_callback(self, callback: Callable[[], None]):
        """设置数据恢复回调函数"""
        self.on_data_recovered_callback = callback

    def udp_initial(self) -> bool:
        try:
            self.socket_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket_udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket_udp.settimeout(1.0)
            self.socket_udp.bind((self.host, self.port))
            self.isconnect = True
            self.udp_running = True
            self.last_receive_time = time.time()
            self.consecutive_timeout_checks = 0
            
            self.udp_thread = Thread(target=self.__udp_process, daemon=True)
            self.udp_thread.start()
            return True
        except socket.error:
            self.isconnect = False
            if self.socket_udp:
                self.socket_udp.close()
            return False

    def __recv(self) -> tuple:
        try:
            data, addr = self.socket_udp.recvfrom(self.buffer_size)
            return data, addr
        except socket.timeout:
            return None, None
        except socket.error:
            return None, None

    def udp_close(self) -> bool:
        self.udp_running = False
        if self.udp_thread and self.udp_thread.is_alive():
            self.udp_thread.join(timeout=1.0)
        time.sleep(0.1)
        if self.socket_udp:
            self.socket_udp.close()
            self.socket_udp = None
        self.isconnect = False
        return True

    def udp_is_connect(self) -> bool:
        return self.isconnect

    def check_timeout(self) -> TimeoutStatus:
        """
        检查超时状态，返回超时状态信息
        外部调用此方法来获取超时状态，并决定如何打印
        """
        current_time = time.time()
        time_since_last_data = current_time - self.last_receive_time
        is_timeout = time_since_last_data > self.no_data_timeout
        
        # 更新连续超时检查次数
        if is_timeout:
            self.consecutive_timeout_checks += 1
        else:
            self.consecutive_timeout_checks = 0
        
        # 更新数据对象的超时状态
        with self.data_lock:
            self.realmocapdata.is_data_timeout = is_timeout
        
        return TimeoutStatus(
            is_timeout=is_timeout,
            last_receive_time=self.last_receive_time,
            time_since_last_data=time_since_last_data,
            frame_index=self.realmocapdata.frame_index,
            timeout_threshold=self.no_data_timeout,
            consecutive_timeout_checks=self.consecutive_timeout_checks
        )

    def is_data_timeout(self) -> bool:
        """快速检查是否超时"""
        current_time = time.time()
        time_since_last_data = current_time - self.last_receive_time
        return time_since_last_data > self.no_data_timeout

    def get_connection_status(self) -> Dict[str, Any]:
        """获取完整的连接状态信息"""
        current_time = time.time()
        time_since_last_data = current_time - self.last_receive_time
        
        with self.data_lock:
            frame_index = self.realmocapdata.frame_index
            is_data_timeout = self.realmocapdata.is_data_timeout
        
        return {
            'is_connected': self.isconnect,
            'is_running': self.udp_running,
            'last_receive_time': self.last_receive_time,
            'last_receive_time_str': datetime.fromtimestamp(self.last_receive_time).strftime('%Y-%m-%d %H:%M:%S') if self.last_receive_time > 0 else '从未',
            'time_since_last_data': time_since_last_data,
            'is_data_timeout': is_data_timeout,
            'timeout_threshold': self.no_data_timeout,
            'frame_index': frame_index,
            'consecutive_timeout_checks': self.consecutive_timeout_checks,
            'device_id': self.device_id,
            'port': self.port
        }

    def __udp_process(self):
        was_timeout = False  # 记录上次检查是否超时
        
        while self.udp_running and self.isconnect:
            try:
                bytes_data, addr = self.__recv()
                
                if bytes_data is not None:
                    # 更新接收时间
                    current_time = time.time()
                    self.last_receive_time = current_time
                    
                    # 检查是否从超时状态恢复
                    if was_timeout:
                        was_timeout = False
                        # 调用数据恢复回调
                        if self.on_data_recovered_callback:
                            self.on_data_recovered_callback()
                    
                    try:
                        json_data = json.loads(bytes_data.decode('utf-8'))
                        with self.data_lock:
                            motion_data = MotionData(json_data)
                            self.realmocapdata.is_update = True
                            self.realmocapdata.last_data_time = current_time
                            self.realmocapdata.frame_index += 1
                            
                            try:
                                l_params = motion_data.list_sequence_params(self.device_id, "l")
                                for name, value in sorted(l_params.items(), key=lambda x: int(x[0][1:])):
                                    if int(name[1:]) >= NODES_HAND: break
                                    self.realmocapdata.jointangle_lHand[int(name[1:])] = np.deg2rad(value) * self.realmocapdata.jointderict_lHand[int(name[1:])]
                            except Exception:
                                pass
                            
                            try:
                                r_params = motion_data.list_sequence_params(self.device_id, "r")
                                for name, value in sorted(r_params.items(), key=lambda x: int(x[0][1:])):
                                    if int(name[1:]) >= NODES_HAND: break
                                    self.realmocapdata.jointangle_rHand[int(name[1:])] = np.deg2rad(value)* self.realmocapdata.jointderict_rHand[int(name[1:])]
                            except Exception:
                                pass
                    
                    except (json.JSONDecodeError, ValueError, Exception):
                        pass
                
                else:
                    # 没有收到数据，检查是否进入超时状态
                    current_timeout_status = self.check_timeout()
                    if current_timeout_status.is_timeout:
                        was_timeout = True
                        # 调用超时回调
                        if self.on_timeout_callback:
                            self.on_timeout_callback(current_timeout_status)
                
            except Exception:
                pass

    def udp_recv_mocap_data(self, mocap_data: UdexRealData) -> bool:
        with self.data_lock:
            mocap_data.frame_index = self.realmocapdata.frame_index
            mocap_data.is_update = self.realmocapdata.is_update
            mocap_data.frequency = self.realmocapdata.frequency
            mocap_data.jointangle_rHand = self.realmocapdata.jointangle_rHand.copy()
            mocap_data.jointangle_lHand = self.realmocapdata.jointangle_lHand.copy()
            mocap_data.last_data_time = self.realmocapdata.last_data_time
            mocap_data.is_data_timeout = self.realmocapdata.is_data_timeout
            
            self.realmocapdata.is_update = False
        
        return True