import socket
import struct
import time
from threading import Thread
import threading

NODES_BODY = 23
NODES_HAND = 20
NODES_FACEBS_ARKIT = 52
NODES_FACEBS_AUDIO = 26

DC_QUAT = 1e-4  # short -> double
DC_POSITION = 1e-3  # short -> double
DC_POWER = 1e-2  # short -> double

uc_ConnectsendBytes = bytes(
    [0xfa, 0x00, 0x00, 0x0b, 0x04, 0x03, 0xa2, 0x53, 0x23, 0x52, 0xce, 0x32, 0x99, 0xf4, 0x32, 0xfb, 0x30])
uc_DisConnectsendBytes = bytes([0xfa, 0x00, 0x00, 0x03, 0x04, 0x0b, 0xa1, 0xfb, 0xa8])


class MocapData:
    def __init__(self):
        self.is_update = False
        self.frame_index = 0
        self.frequency = 0
        self.ns_result = 0

        self.sensor_state_body = [0] * NODES_BODY
        self.position_body = [[0.0, 0.0, 0.0] for _ in range(NODES_BODY)]
        self.quaternion_body = [[0.0, 0.0, 0.0, 0.0] for _ in range(NODES_BODY)]
        self.gyr_body = [[0.0, 0.0, 0.0] for _ in range(NODES_BODY)]
        self.acc_body = [[0.0, 0.0, 0.0] for _ in range(NODES_BODY)]
        self.velocity_body = [[0.0, 0.0, 0.0] for _ in range(NODES_BODY)]

        self.sensor_state_r_hand = [0] * NODES_HAND
        self.position_rHand = [[0.0, 0.0, 0.0] for _ in range(NODES_HAND)]
        self.quaternion_rHand = [[0.0, 0.0, 0.0, 0.0] for _ in range(NODES_HAND)]
        self.gyr_r_hand = [[0.0, 0.0, 0.0] for _ in range(NODES_HAND)]
        self.acc_r_hand = [[0.0, 0.0, 0.0] for _ in range(NODES_HAND)]
        self.velocity_r_hand = [[0.0, 0.0, 0.0] for _ in range(NODES_HAND)]

        self.sensor_state_l_hand = [0] * NODES_HAND
        self.position_lHand = [[0.0, 0.0, 0.0] for _ in range(NODES_HAND)]
        self.quaternion_lHand = [[0.0, 0.0, 0.0, 0.0] for _ in range(NODES_HAND)]
        self.gyr_l_hand = [[0.0, 0.0, 0.0] for _ in range(NODES_HAND)]
        self.acc_l_hand = [[0.0, 0.0, 0.0] for _ in range(NODES_HAND)]
        self.velocity_l_hand = [[0.0, 0.0, 0.0] for _ in range(NODES_HAND)]

        self.is_use_face_blend_shapes_arkit = False
        self.is_use_face_blend_shapes_audio = False
        self.face_blend_shapes_arkit = [0.0] * NODES_FACEBS_ARKIT
        self.face_blend_shapes_audio = [0.0] * NODES_FACEBS_AUDIO
        self.local_quat_right_eyeball = [0.0] * 4
        self.local_quat_left_eyeball = [0.0] * 4


class VtrdynSocketUdp:
    def __init__(self, debug = False):
        self.socket_udp = None
        self.is_use_face_blend_shapes_arkit = False
        self.send_thread = None
        self.send_running = False
        self.mocap_data_realtime = MocapData()
        self.data_lock = threading.Lock()
        self.isconnect = False
        self.debug = debug

    def udp_initial(self, local_port: int) -> bool:
        """初始化 UDP socket"""
        try:
            self.socket_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket_udp.bind(('', local_port))
            self.socket_udp.settimeout(10)
            self.isconnect = True
            return True
        except socket.error as e:
            self.isconnect = False
            if self.debug:
                print(f"Socket initialization failed: {e}")
            if self.socket_udp:
                self.socket_udp.close()
            return False

    @staticmethod
    def udp_getsockaddr(ip: str, port: int) -> tuple:
        """将 IP 地址及端口号转化为能识别的地址格式"""
        return (ip, port)

    def __recv(self, buffer_size: int = 3415) -> tuple:
        try:
            data, addr = self.socket_udp.recvfrom(buffer_size)
            return data, addr
        except socket.error as e:
            return None, None

    def udp_close(self, dst_addr: tuple) -> bool:
        self.socket_udp.sendto(uc_DisConnectsendBytes, dst_addr)
        self.send_running = False
        self.send_thread.join()
        time.sleep(0.1)
        if self.socket_udp:
            self.socket_udp.close()
        return self.send_running

    def udp_send_request_connect(self, dst_addr: tuple) -> bool:
        connerrflag = False
        try:
            self.socket_udp.sendto(uc_ConnectsendBytes, dst_addr)
            if self.debug:
                print(f"Initialization packet sent to {dst_addr}")
            # 等待确认响应
            bytes_data, addr = self.socket_udp.recvfrom(1024)
            if addr == dst_addr:
                connerrflag = True
                if self.debug:
                    print(f"Connection established with {addr}")
        except socket.timeout:
            if self.debug:
                print("Initialization timeout: No response from target")
        except socket.error as e:
            if self.debug:
                print(f"Initialization error: {str(e)}")
        self.send_running = True
        self.send_thread = Thread(target=self.__udp_process)
        self.send_thread.start()
        return connerrflag

    def udp_is_onnect(self) -> bool:
        return self.isconnect

    def __udp_process(self):
        while self.send_running:
            try:
                bytes_data, addr = self.__recv()
                if bytes_data is None:
                    self.isconnect = False
                    time.sleep(0.01)
                    continue
                self.isconnect = True
                if len(bytes_data) < 683 or (bytes_data[2] << 8 | bytes_data[3]) - 3 < (NODES_BODY * 8) or bytes_data[
                    0] != 250 or bytes_data[681] != 251:
                    return False
                mocap_temp = MocapData()
                offset = 1
                mocap_temp.frame_index = bytes_data[offset]
                offset = 7  # Move to the next field
                mocap_temp.is_update = bool(bytes_data[offset])
                offset = 10  # Move to frequency
                mocap_temp.frequency = bytes_data[offset]
                offset = 11  # Move to hips_position
                for i in range(3):
                    # 每个位置分量
                    mocap_temp.position_body[i] = \
                    struct.unpack('>h', bytes_data[offset + i * 2:offset + i * 2 + 2])[
                        0] * DC_POSITION
                offset = offset + 6 + NODES_BODY  # 移动到 quaternion_body 开始的位置
                for i in range(NODES_BODY):
                    for j in range(4):
                        current_offset = offset + i * 8 + j * 2
                        mocap_temp.quaternion_body[i][j] = \
                            struct.unpack('>h', bytes_data[current_offset:current_offset + 2])[
                                0] * DC_QUAT
                offset = offset + NODES_BODY * 8 + NODES_HAND  # 移动到 quaternion_rightHand 开始的位置
                for i in range(NODES_HAND):
                    for j in range(4):
                        current_offset = offset + i * 8 + j * 2
                        mocap_temp.quaternion_rHand[i][j] = \
                            struct.unpack('>h', bytes_data[current_offset:current_offset + 2])[
                                0] * DC_QUAT
                offset = offset + NODES_HAND * 8 + NODES_HAND  # 移动到 quaternion_leftHand 开始的位置
                for i in range(NODES_HAND):
                    for j in range(4):
                        current_offset = offset + i * 8 + j * 2
                        mocap_temp.quaternion_lHand[i][j] = \
                            struct.unpack('>h', bytes_data[current_offset:current_offset + 2])[
                                0] * DC_QUAT
                offset = offset + NODES_HAND * 8 + 1  # 移动到 isUseBlendShapeArkit
                mocap_temp.is_use_face_blend_shapes_arkit = bool(bytes_data[offset])
                with self.data_lock:
                    self.mocap_data_realtime.frame_index = mocap_temp.frame_index

                    self.mocap_data_realtime.is_update = mocap_temp.is_update
                    self.mocap_data_realtime.frequency = mocap_temp.frequency
                    self.mocap_data_realtime.quaternion_rHand = mocap_temp.quaternion_rHand
                    self.mocap_data_realtime.quaternion_lHand = mocap_temp.quaternion_lHand
                time.sleep(0.01)
            except socket.timeout:
                # 检查连接超时
                if self.connected:
                    if self.debug:
                        print("Connection timeout detected!")
                    self.connected = False
                    break
                continue
                
            except Exception as e:
                if self.debug:
                    print(f"Receive error: {str(e)}")
                break

    def udp_recv_mocap_data(self, mocap_data: MocapData) -> bool:
        with self.data_lock:
            mocap_data.frame_index = self.mocap_data_realtime.frame_index
            mocap_data.is_update = self.mocap_data_realtime.is_update
            mocap_data.frequency = self.mocap_data_realtime.frequency
            mocap_data.quaternion_rHand = self.mocap_data_realtime.quaternion_rHand
            mocap_data.quaternion_lHand = self.mocap_data_realtime.quaternion_lHand
        return True
