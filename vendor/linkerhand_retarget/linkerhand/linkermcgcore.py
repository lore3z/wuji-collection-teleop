from datetime import datetime
import socket
import time
from threading import Thread
import json
from dataclasses import dataclass
from typing import List, Dict, Union, Any
import threading
import numpy as np

NODES_HAND = 25

LOG_FILE_PATH = "/tmp/a.log"

@dataclass
class HandData:
    pitch: List[int]      # 5个手指的pitch值 [0-255]
    side: List[int]       # 5个手指的side值 [0-255]
    roll: List[int]       # 5个手指的roll值 [0-255]
    two_pitch: List[int]  # 5个手指的two_pitch值 [0-255]
    end_pitch: List[int]  # 5个手指的end_pitch值 [0-255]


class HaoCunData:
    def __init__(self):
        self.is_update = False
        self.frame_index = 0
        self.frequency = 0
        self.ns_result = 0
        # 直接存储25个字节值，不需要转换
        self.jointangle_rHand = [0.0] * NODES_HAND
        self.jointangle_lHand = [0.0] * NODES_HAND


class HaoCunScoketUdp:
    def __init__(self, host='127.0.0.1', port=7000, buffer_size=2048):
        """
        初始化UDP客户端
        
        Args:
            target_host: 目标服务器地址
            target_port: 目标服务器端口
            buffer_size: 缓冲区大小
            device_id: 设备ID
        """
        self.socket_udp = None
        self.is_use_face_blend_shapes_arkit = False
        self.udp_thread = None
        self.udp_running = False
        self.isconnect = False
        self.target_host = host
        self.target_port = port
        self.target_address = (host, port)
        self.buffer_size = buffer_size
        self.realmocapdata = HaoCunData()
        self.data_lock = threading.Lock()
        self.frame_counter = 0

    def udp_initial(self) -> bool:
        """初始化UDP客户端并连接到目标服务器"""
        try:
            # 创建UDP socket
            self.socket_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # 设置超时时间
            self.socket_udp.settimeout(10)
            
            # UDP是面向无连接的，这里只是保存目标地址，不会真正建立连接
            # 但我们可以发送一个测试包来验证连通性
            try:
                test_packet = b"CONNECT"
                self.socket_udp.sendto(test_packet, self.target_address)
                self.socket_udp.settimeout(2)  # 设置较短的超时用于连接测试
                # 尝试接收响应（如果服务器会响应的话）
                # 注意：某些UDP服务可能不会响应，这并不代表连接失败
                try:
                    data, addr = self.socket_udp.recvfrom(self.buffer_size)
                    print(f"成功连接到服务器 {addr}")
                except socket.timeout:
                    print(f"已发送连接请求到 {self.target_address} (UDP协议，无连接确认)")
            except Exception as e:
                print(f"连接测试时出错: {e}")
            
            # 恢复超时设置
            self.socket_udp.settimeout(10)
            
            self.isconnect = True
            self.udp_running = True
            self.udp_thread = Thread(target=self.__udp_process)
            self.udp_thread.start()
            return True
        except socket.error as e:
            self.isconnect = False
            print(f"UDP客户端初始化错误: {e}")
            if self.socket_udp:
                self.socket_udp.close()
            return False

    def __recv(self) -> tuple:
        """接收UDP数据"""
        try:
            data, addr = self.socket_udp.recvfrom(self.buffer_size)
            return data, addr
        except socket.timeout:
            return None, None
        except socket.error as e:
            print(f"接收数据时出错: {e}")
            return None, None

    def udp_close(self) -> bool:
        """关闭UDP连接"""
        self.udp_running = False
        if self.udp_thread and self.udp_thread.is_alive():
            self.udp_thread.join(timeout=2)
        time.sleep(0.1)
        if self.socket_udp:
            self.socket_udp.close()
        self.isconnect = False
        return True

    def udp_is_connect(self) -> bool:
        """检查连接状态"""
        return self.isconnect

    def __udp_process(self):
        """UDP数据处理线程"""
        errorprintcount = 0
        while self.udp_running:
            try:
                bytes_data, addr = self.__recv()
                if bytes_data is not None:
                    try:
                        # 将接收到的数据传递给处理函数
                        self.__process_received_data(bytes_data)
                    except Exception as e:
                        if errorprintcount > 100:
                            print(f"数据处理错误: {e}")
                            errorprintcount = 0
                # 添加短暂休眠避免CPU占用过高
                time.sleep(0.001)
            except Exception as e:
                if errorprintcount > 100:
                    print(f"UDP处理线程错误: {e}")
                    errorprintcount = 0
            finally:
                errorprintcount += 1

    def __process_received_data(self, bytes_data: bytes):
        """处理接收到的数据"""
        try:
            # 解码JSON数据
            json_str = bytes_data.decode('utf-8', errors='replace')
            data = json.loads(json_str)
            
            # 更新帧计数器
            self.frame_counter += 1
            
            # 处理数据
            with self.data_lock:
                # 处理左手数据
                if 'leftHand' in data:
                    left_hand = data['leftHand']
                    self.realmocapdata.jointangle_lHand = self.extract_25_bytes(left_hand)
                
                # 处理右手数据
                if 'rightHand' in data:
                    right_hand = data['rightHand']
                    self.realmocapdata.jointangle_rHand = self.extract_25_bytes(right_hand)
                
                # 更新其他状态
                self.realmocapdata.is_update = True
                self.realmocapdata.frame_index = self.frame_counter
                
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始数据: {bytes_data.decode('utf-8', errors='replace')}")
        except Exception as e:
            print(f"处理数据时出错: {e}")

    def extract_25_bytes(self, hand_data: Dict) -> List[float]:
        """
        从手部数据字典中提取25个字节值，按以下顺序排列：
        1. pitch (5个值)
        2. side (5个值) 
        3. roll (5个值)
        4. two_pitch (5个值)
        5. end_pitch (5个值)
        
        总计25个值
        
        Args:
            hand_data: 包含pitch, side, roll, two_pitch, end_pitch的字典
            
        Returns:
            长度为25的字节值列表
        """
        byte_values = [0.0] * NODES_HAND
        idx = 0
        
        try:
            # 按顺序提取5个数组，每个5个值，共25个值
            arrays_to_extract = ['pitch', 'side', 'roll', 'two_pitch', 'end_pitch']
            
            for array_name in arrays_to_extract:
                if array_name in hand_data:
                    values = hand_data[array_name]
                    # 确保有5个值
                    if len(values) >= 5:
                        for i in range(5):
                            if idx < NODES_HAND:
                                byte_values[idx] = float(values[i])
                                idx += 1
                    else:
                        # 如果数据不足5个，填充0
                        for i in range(5):
                            if idx < NODES_HAND:
                                byte_values[idx] = 0.0
                                idx += 1
                else:
                    # 如果缺少某个数组，填充5个0
                    for i in range(5):
                        if idx < NODES_HAND:
                            byte_values[idx] = 0.0
                            idx += 1
                            
        except Exception as e:
            print(f"提取字节值时出错: {e}")
            
        return byte_values

    def udp_recv_mocap_data(self, mocap_data: HaoCunData) -> bool:
        """获取最新的动捕数据"""
        with self.data_lock:
            mocap_data.frame_index = self.realmocapdata.frame_index
            mocap_data.is_update = self.realmocapdata.is_update
            mocap_data.frequency = self.realmocapdata.frequency
            mocap_data.jointangle_rHand = self.realmocapdata.jointangle_rHand.copy()
            mocap_data.jointangle_lHand = self.realmocapdata.jointangle_lHand.copy()
        return True

    def send_data(self, data: bytes) -> bool:
        """发送数据到目标服务器"""
        try:
            if not self.isconnect or not self.socket_udp:
                print("UDP客户端未连接")
                return False
            
            self.socket_udp.sendto(data, self.target_address)
            return True
        except Exception as e:
            print(f"发送数据时出错: {e}")
            return False

    def get_hand_data_summary(self) -> Dict:
        """获取手部数据摘要"""
        with self.data_lock:
            return {
                'frame_index': self.realmocapdata.frame_index,
                'is_update': self.realmocapdata.is_update,
                'left_hand_first_5': self.realmocapdata.jointangle_lHand[:5],
                'right_hand_first_5': self.realmocapdata.jointangle_rHand[:5],
                'left_hand_total': len(self.realmocapdata.jointangle_lHand),
                'right_hand_total': len(self.realmocapdata.jointangle_rHand)
            }
