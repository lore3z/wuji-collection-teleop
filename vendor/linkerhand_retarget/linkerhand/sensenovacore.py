from datetime import datetime
import socket
import numpy as np
import time
from threading import Thread
import json
from dataclasses import dataclass
from typing import List, Dict, Union, Any
import threading

NODES_HAND = 30

json_send_basic = {
    "timsstamp": "2025-4-30 22:47:90.123",
    "datatype": "datarecv",
    "right": {
        "thumb": {
            "normalforce": 0.0,
            "approachforce": 0.0,
            "tangentialforce": 0.0
        },
        "index": {
            "normalforce": 0.0,
            "approachforce": 0.0,
            "tangentialforce": 0.0
        },
        "middle": {
            "normalforce": 0.0,
            "approachforce": 0.0,
            "tangentialforce": 0.0
        },
        "ring": {
            "normalforce": 31.0,
            "approachforce": 32.0,
            "tangentialforce": 0.0
        },
        "pinky": {
            "normalforce": 0.0,
            "approachforce": 0.0,
            "tangentialforce": 0.0
        }
    },
    "left": {
        "thumb": {
            "normalforce": 0.0,
            "approachforce": 0.0,
            "tangentialforce": 0.0
        },
        "index": {
            "normalforce": 0.0,
            "approachforce": 0.0,
            "tangentialforce": 0.0
        },
        "middle": {
            "normalforce": 0.0,
            "approachforce": 0.0,
            "tangentialforce": 0.0
        },
        "ring": {
            "normalforce": 0.0,
            "approachforce": 0.0,
            "tangentialforce": 0.0
        },
        "pinky": {
            "normalforce": 0.0,
            "approachforce": 0.0,
            "tangentialforce": 0.0
        }
    }
}


class SenseNovaData:
    def __init__(self):
        self.is_update = False
        self.frame_index = 0
        self.frequency = 0
        self.ns_result = 0
        self.jointangle_rHand = [0.0] * NODES_HAND
        self.jointangle_lHand = [0.0] * NODES_HAND
        self.normalforce_rHand = [0.0] * 5
        self.normalforce_lHand = [0.0] * 5
        self.approachforce_rHand = [0.0] * 5
        self.approachforce_lHand = [0.0] * 5


class SenseNovaScoketUdp:
    def __init__(self, host='0.0.0.0', port=7000, buffer_size=4098):
        self.socket_udp = None
        self.is_use_face_blend_shapes_arkit = False
        self.udp_thread = None
        self.udp_running = False
        self.isconnect = False
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.udp_addr = self.udp_getsockaddr(host, port)
        self.realmocapdata = SenseNovaData()
        self.data_lock = threading.Lock()

    def udp_initial(self) -> bool:
        """初始化 UDP socket"""
        try:
            self.socket_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket_udp.bind(('', 8888))
            self.socket_udp.settimeout(10)
            self.isconnect = True
            self.udp_running = True
            self.udp_thread = Thread(target=self.__udp_process)
            self.udp_thread.start()

            return True
        except socket.error as e:
            self.isconnect = False
            print(f"发生错误: {e},UDP套接字已关闭!")
            if self.socket_udp:
                self.socket_udp.close()
            return False

    @staticmethod
    def udp_getsockaddr(ip: str, port: int) -> tuple:
        """将 IP 地址及端口号转化为能识别的地址格式"""
        return (ip, port)

    def __send(self):
        try:
            json_send_basic["right"]["thumb"]["normalforce"] = self.realmocapdata.normalforce_rHand[0]
            json_send_basic["right"]["index"]["normalforce"] = self.realmocapdata.normalforce_rHand[1]
            json_send_basic["right"]["middle"]["normalforce"] = self.realmocapdata.normalforce_rHand[2]
            json_send_basic["right"]["ring"]["normalforce"] = self.realmocapdata.normalforce_rHand[3]
            json_send_basic["right"]["pinky"]["normalforce"] = self.realmocapdata.normalforce_rHand[4]
            json_send_basic["left"]["thumb"]["normalforce"] = self.realmocapdata.normalforce_lHand[0]
            json_send_basic["left"]["index"]["normalforce"] = self.realmocapdata.normalforce_lHand[1]
            json_send_basic["left"]["middle"]["normalforce"] = self.realmocapdata.normalforce_lHand[2]
            json_send_basic["left"]["ring"]["normalforce"] = self.realmocapdata.normalforce_lHand[3]
            json_send_basic["left"]["pinky"]["normalforce"] = self.realmocapdata.normalforce_lHand[4]
            json_send_basic["right"]["thumb"]["approachforce"] = self.realmocapdata.approachforce_rHand[0]
            json_send_basic["right"]["index"]["approachforce"] = self.realmocapdata.approachforce_rHand[1]
            json_send_basic["right"]["middle"]["approachforce"] = self.realmocapdata.approachforce_rHand[2]
            json_send_basic["right"]["ring"]["approachforce"] = self.realmocapdata.approachforce_rHand[3]
            json_send_basic["right"]["pinky"]["approachforce"] = self.realmocapdata.approachforce_rHand[4]
            json_send_basic["left"]["thumb"]["approachforce"] = self.realmocapdata.approachforce_lHand[0]
            json_send_basic["left"]["index"]["approachforce"] = self.realmocapdata.approachforce_lHand[1]
            json_send_basic["left"]["middle"]["approachforce"] = self.realmocapdata.approachforce_lHand[2]
            json_send_basic["left"]["ring"]["approachforce"] = self.realmocapdata.approachforce_lHand[3]
            json_send_basic["left"]["pinky"]["approachforce"] = self.realmocapdata.approachforce_lHand[4]
            json_data = json.dumps(json_send_basic)
            self.socket_udp.sendto(json_data.encode('utf-8'), self.udp_addr)
        except Exception as e:
            print(f"Send未知错误: {e}")
        finally:
            pass

    def __recv(self) -> tuple:
        try:
            data, addr = self.socket_udp.recvfrom(self.buffer_size)
            return data, addr
        except socket.timeout:
            return None, None

    def udp_close(self) -> bool:
        self.udp_running = False
        self.udp_thread.join()
        time.sleep(0.1)
        if self.socket_udp:
            self.socket_udp.close()
        return True

    def udp_is_onnect(self) -> bool:
        return self.isconnect

    def __udp_process(self):
        errorprintcount = 0
        while self.udp_running:
            self.__send()
            try:
                bytes_data, addr = self.__recv()
                if bytes_data is not None:
                    try:
                        print(bytes_data)
                        json_data = json.loads(bytes_data.decode('utf-8'))
                        self.realmocapdata.jointangle_rHand[0:6] = [
                            json_data["euler"]["right"]["thumb"]["cmc_roll"],
                            json_data["euler"]["right"]["thumb"]["cmc_yaw"],
                            json_data["euler"]["right"]["thumb"]["cmc_pitch"],
                            json_data["euler"]["right"]["thumb"]["mcp"],
                            json_data["euler"]["right"]["thumb"]["pip"],
                            json_data["euler"]["right"]["thumb"]["dip"]]
                        self.realmocapdata.jointangle_rHand[6:12] = [
                            json_data["euler"]["right"]["index"]["cmc_roll"],
                            json_data["euler"]["right"]["index"]["cmc_yaw"],
                            json_data["euler"]["right"]["index"]["cmc_pitch"],
                            json_data["euler"]["right"]["index"]["mcp"],
                            json_data["euler"]["right"]["index"]["pip"],
                            json_data["euler"]["right"]["index"]["dip"]]
                        self.realmocapdata.jointangle_rHand[12:18] = [
                            json_data["euler"]["right"]["middle"]["cmc_roll"],
                            json_data["euler"]["right"]["middle"]["cmc_yaw"],
                            json_data["euler"]["right"]["middle"]["cmc_pitch"],
                            json_data["euler"]["right"]["middle"]["mcp"],
                            json_data["euler"]["right"]["middle"]["pip"],
                            json_data["euler"]["right"]["middle"]["dip"]]
                        self.realmocapdata.jointangle_rHand[18:24] = [
                            json_data["euler"]["right"]["ring"]["cmc_roll"],
                            json_data["euler"]["right"]["ring"]["cmc_yaw"],
                            json_data["euler"]["right"]["ring"]["cmc_pitch"],
                            json_data["euler"]["right"]["ring"]["mcp"],
                            json_data["euler"]["right"]["ring"]["pip"],
                            json_data["euler"]["right"]["ring"]["dip"]]
                        self.realmocapdata.jointangle_rHand[24:30] = [
                            json_data["euler"]["right"]["pinky"]["cmc_roll"],
                            json_data["euler"]["right"]["pinky"]["cmc_yaw"],
                            json_data["euler"]["right"]["pinky"]["cmc_pitch"],
                            json_data["euler"]["right"]["pinky"]["mcp"],
                            json_data["euler"]["right"]["pinky"]["pip"],
                            json_data["euler"]["right"]["pinky"]["dip"]]
                        self.realmocapdata.jointangle_lHand[0:6] = [
                            json_data["euler"]["left"]["thumb"]["cmc_roll"],
                            json_data["euler"]["left"]["thumb"]["cmc_yaw"],
                            json_data["euler"]["left"]["thumb"]["cmc_pitch"],
                            json_data["euler"]["left"]["thumb"]["mcp"],
                            json_data["euler"]["left"]["thumb"]["pip"],
                            json_data["euler"]["left"]["thumb"]["dip"]]
                        self.realmocapdata.jointangle_lHand[6:12] = [
                            json_data["euler"]["left"]["index"]["cmc_roll"],
                            json_data["euler"]["left"]["index"]["cmc_yaw"],
                            json_data["euler"]["left"]["index"]["cmc_pitch"],
                            json_data["euler"]["left"]["index"]["mcp"],
                            json_data["euler"]["left"]["index"]["pip"],
                            json_data["euler"]["left"]["index"]["dip"]]
                        self.realmocapdata.jointangle_lHand[12:18] = [
                            json_data["euler"]["left"]["middle"]["cmc_roll"],
                            json_data["euler"]["left"]["middle"]["cmc_yaw"],
                            json_data["euler"]["left"]["middle"]["cmc_pitch"],
                            json_data["euler"]["left"]["middle"]["mcp"],
                            json_data["euler"]["left"]["middle"]["pip"],
                            json_data["euler"]["left"]["middle"]["dip"]]
                        self.realmocapdata.jointangle_lHand[18:24] = [
                            json_data["euler"]["left"]["ring"]["cmc_roll"],
                            json_data["euler"]["left"]["ring"]["cmc_yaw"],
                            json_data["euler"]["left"]["ring"]["cmc_pitch"],
                            json_data["euler"]["left"]["ring"]["mcp"],
                            json_data["euler"]["left"]["ring"]["pip"],
                            json_data["euler"]["left"]["ring"]["dip"]]
                        self.realmocapdata.jointangle_lHand[24:30] = [
                            json_data["euler"]["left"]["pinky"]["cmc_roll"],
                            json_data["euler"]["left"]["pinky"]["cmc_yaw"],
                            json_data["euler"]["left"]["pinky"]["cmc_pitch"],
                            json_data["euler"]["left"]["pinky"]["mcp"],
                            json_data["euler"]["left"]["pinky"]["pip"],
                            json_data["euler"]["left"]["pinky"]["dip"]]
                    except json.JSONDecodeError as e:
                        if errorprintcount > 100:
                            print(f"JSON解析错误: {e}")
                            print(f"原始数据: {bytes_data.decode('utf-8', errors='replace')}")
                            errorprintcount = 0
                    except ValueError as e:  # 新增：捕获 ValueError
                        if errorprintcount > 100:
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 设备ID错误: {e}")
                            errorprintcount = 0
            except Exception as e:
                if errorprintcount > 100:
                    if e.args[0] == 10054:
                        print("远程设备已经断开!")
                    else:
                        print(f"Recv未知错误: {e}")
                    errorprintcount = 0
            finally:
                errorprintcount += 1
                time.sleep(0.001)
