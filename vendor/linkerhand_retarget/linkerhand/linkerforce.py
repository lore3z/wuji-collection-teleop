import array
import threading
import numpy as np
import time
import re
import struct
import serial
import serial.tools.list_ports
from threading import Thread, Event
from enum import Enum
from typing import List, Dict, Optional, Callable, Any, Set
from .constants import HandType


# ============== 常量定义 ==============

class CommandCode(Enum):
    VERSION_QUERY = 0x01
    SET_FLAG = 0x02
    POSITION_QUERY = 0x03
    FORCE_FEEDBACK = 0x04
    A3_POSITION = 0xA3
    A6_POSITION = 0xA6
    A7_FORCE = 0xA7


# 协议常量
BUFFER_SIZE = 1024
MAX_FRAME_DATA_SIZE = 255
FRAME_HEADER = 0x5D

# 时序常量
WARMUP_DELAY = 0.15
RESPONSE_WAIT = 0.3
FINAL_WAIT = 1.0
READ_INTERVAL = 0.003
ERROR_DELAY = 0.01
RETRY_DELAY = 0.5
QUERY_INTERVAL = 10
VERSION_QUERY_RETRY_COUNT = 10

# 超时常量
CONNECTION_TIMEOUT = 5.0
CHECK_INTERVAL = 5.0

# USB 设备匹配模式
USB_PATTERNS = [
    r'/dev/ttyUSB\d+',
    r'/dev/ttyACM\d+',
    r'/dev/ttyXRUSB\d+',
    r'/dev/ttyOBC\d+',
]


# ============== 环形缓冲区 ==============

class CircularBuffer:
    def __init__(self):
        self.data = array.array('B', [0] * BUFFER_SIZE)
        self.read_pos = 0
        self.write_pos = 0
        self.data_len = 0

    def write(self, data):
        for byte in data:
            self.data[self.write_pos] = byte
            self.write_pos = (self.write_pos + 1) % BUFFER_SIZE
            if self.data_len < BUFFER_SIZE:
                self.data_len += 1
            else:
                self.read_pos = (self.read_pos + 1) % BUFFER_SIZE

    def read_byte(self):
        if self.data_len == 0:
            return None
        byte = self.data[self.read_pos]
        self.read_pos = (self.read_pos + 1) % BUFFER_SIZE
        self.data_len -= 1
        return byte


# ============== 帧解析器 ==============

class FrameParseState(Enum):
    HEADER = 0
    CMD = 1
    LENGTH = 2
    DATA = 3
    CHECKSUM = 4


class FrameParser:
    def __init__(self):
        self.state = FrameParseState.HEADER
        self.frame_buf = array.array('B', [0] * (3 + MAX_FRAME_DATA_SIZE + 1))
        self.expected_len = 0
        self.current_pos = 0
        self.checksum = 0

    def reset(self):
        self.state = FrameParseState.HEADER
        self.current_pos = 0
        self.checksum = 0
        for i in range(len(self.frame_buf)):
            self.frame_buf[i] = 0

    def process_byte(self, byte):
        byte = byte & 0xFF
        if self.state == FrameParseState.HEADER:
            if byte == FRAME_HEADER:
                self.frame_buf[0] = byte
                self.current_pos = 1
                self.checksum = 0
                self.state = FrameParseState.CMD
        elif self.state == FrameParseState.CMD:
            self.frame_buf[1] = byte
            self.current_pos = 2
            self.state = FrameParseState.LENGTH
        elif self.state == FrameParseState.LENGTH:
            self.frame_buf[2] = byte
            self.expected_len = 3 + byte + 1
            self.current_pos = 3
            if 0 < byte <= MAX_FRAME_DATA_SIZE:
                self.state = FrameParseState.DATA
            else:
                self.state = FrameParseState.CHECKSUM
        elif self.state == FrameParseState.DATA:
            self.frame_buf[self.current_pos] = byte
            self.current_pos += 1
            if self.current_pos >= self.expected_len - 1:
                self.state = FrameParseState.CHECKSUM
        elif self.state == FrameParseState.CHECKSUM:
            if self.checksum == byte:
                self.frame_buf[self.current_pos] = byte
                return True
            else:
                self.reset()

        if self.state != FrameParseState.HEADER:
            self.checksum = (self.checksum + byte) & 0xFF
        
        return False


# ============== 日志工具 ==============

class Logger:
    def __init__(self, logger_func: Optional[Callable[[str, str], None]] = None, isdebug: bool = False):
        self.logger = logger_func
        self.isdebug = isdebug

    def log(self, level: str, msg: str) -> None:
        if self.logger:
            if self.isdebug and level == 'debug':
                self.logger('info', msg)
            else:
                self.logger(level, msg)
        else:
            print(msg)


# ============== 串口扫描器 ==============

class SerialScanner:
    def __init__(self, baudrates: Optional[List[int]] = None, 
                 exclude_ports: Optional[List[str]] = None, 
                 logger: Optional[Logger] = None):
        self.baudrates = baudrates or [2000000, 1000000, 921600, 460800]
        self.exclude_ports = set(exclude_ports) if exclude_ports else set()
        self.checked_ports: Set[str] = set()
        self.logger = logger

    def is_usb_device(self, port_name):
        for pattern in USB_PATTERNS:
            if re.match(pattern, port_name):
                return True
        try:
            ports = serial.tools.list_ports.comports()
            for port_info in ports:
                if port_info.device == port_name:
                    description = (port_info.description or "").lower()
                    if any(kw in description for kw in ['usb', 'serial', 'com']):
                        return True
                    if port_info.hwid and 'USB' in port_info.hwid.upper():
                        return True
        except:
            pass
        return False

    def scan_available_ports(self):
        ports = serial.tools.list_ports.comports()
        available = []
        for port in ports:
            device = port.device
            if not self.is_usb_device(device):
                continue
            if device in self.exclude_ports:
                if self.logger:
                    self.logger.log('debug', f"跳过排除的串口: {device}")
                continue
            if device not in self.checked_ports:
                available.append(device)
        return available


# ============== 帧处理器 ==============

class FrameHandler:
    def __init__(self, handtype: HandType, logger: Optional['Logger'] = None):
        self._handtype = handtype  # 期望的手类型
        self.logger = logger
        self._data_lock = threading.Lock()
        self._poslist: List[float] = [0.0] * 21
        self._forcelist: List[float] = [0.0] * 5
        self._realforcelist: List[int] = [0] * 5

    @property
    def poslist(self) -> List[float]:
        with self._data_lock:
            return self._poslist.copy()

    @poslist.setter
    def poslist(self, value: List[float]):
        with self._data_lock:
            self._poslist = value

    @property
    def forcelist(self) -> List[float]:
        with self._data_lock:
            return self._forcelist.copy()

    @forcelist.setter
    def forcelist(self, value: List[float]):
        with self._data_lock:
            self._forcelist = value

    @property
    def realforcelist(self) -> List[int]:
        with self._data_lock:
            return self._realforcelist.copy()

    @realforcelist.setter
    def realforcelist(self, value: List[int]):
        with self._data_lock:
            self._realforcelist = value

    def handle_frame(self, frame: array.array) -> Optional[Dict[str, Any]]:
        cmd = frame[1]
        data_len = frame[2]
        frame_data = frame[3:3 + data_len]

        if cmd == CommandCode.VERSION_QUERY.value:
            return self._handle_version(frame_data)
        elif cmd == CommandCode.POSITION_QUERY.value:
            return self._handle_position(frame_data, is_a3=False)
        elif cmd == CommandCode.FORCE_FEEDBACK.value:
            return self._handle_force(frame_data)
        elif cmd == CommandCode.A3_POSITION.value:
            return self._handle_position(frame_data, is_a3=True)
        elif cmd == CommandCode.A6_POSITION.value:
            return self._handle_a6_position(frame_data)
        else:
            if self.logger:
                self.logger.log('warn', f"Unknown command: 0x{cmd:02X}")
            return None

    def _handle_version(self, frame_data: array.array) -> Dict[str, Any]:
        value = struct.unpack('<I', frame_data[:4])[0]
        value_str = f"{value:05d}"
        major = int(value_str[0])
        minor = int(value_str[1:3])
        sub = int(value_str[3:5])
        version = f"{major}.{minor}.{sub}"
        status_code = frame_data[4]
        detected_type = self._get_hand_type(status_code)
        raw_type = self.detect_hand_type(status_code)
        return {'version': version, 'handtype': detected_type, 'raw_handtype': raw_type}

    def _handle_position(self, frame_data: array.array, is_a3: bool = False) -> Optional[Dict[str, Any]]:
        if len(frame_data) % 4 != 0:
            if self.logger:
                self.logger.log('warn', f"Invalid position data length: {len(frame_data)}")
            return None
        floats: List[float] = []
        for i in range(len(frame_data) // 4):
            try:
                val = struct.unpack('<f', frame_data[i*4:(i+1)*4])[0]
                floats.append(np.deg2rad(val))
            except struct.error as e:
                if self.logger:
                    self.logger.log('warn', f"Unpack error: {e}")
        self.poslist = floats
        result: Dict[str, Any] = {'poslist': floats}
        if is_a3:
            result['a6count'] = True
        return result

    def _handle_force(self, frame_data: array.array) -> Optional[Dict[str, Any]]:
        if len(frame_data) % 2 != 0:
            if self.logger:
                self.logger.log('warn', f"Invalid force data length: {len(frame_data)}")
            return None
        values: List[int] = []
        for i in range(len(frame_data) // 2):
            try:
                val = struct.unpack('>h', frame_data[i*2:(i+1)*2])[0]
                values.append(val)
            except struct.error as e:
                if self.logger:
                    self.logger.log('warn', f"Unpack error: {e}")
        self.realforcelist = values
        return {'realforcelist': values}

    def _handle_a6_position(self, frame_data: array.array) -> Optional[Dict[str, Any]]:
        if len(frame_data) % 2 != 0:
            if self.logger:
                self.logger.log('warn', f"Invalid a6 data length: {len(frame_data)}")
            return None
        floats: List[float] = []
        for i in range(len(frame_data) // 2):
            try:
                val = struct.unpack('<h', frame_data[i*2:(i+1)*2])[0]
                floats.append(np.deg2rad(val / 100))
            except struct.error as e:
                if self.logger:
                    self.logger.log('warn', f"Unpack error: {e}")
        self.poslist = floats
        return {'poslist': floats, 'force_response': True}

    def _get_hand_type(self, status_code: int) -> Optional[str]:
        """返回手类型字符串，保持向后兼容"""
        if status_code == 0 and self._handtype == HandType.left:
            return "Left"
        elif status_code == 1 and self._handtype == HandType.right:
            return "Right"
        return None
    
    def detect_hand_type(self, status_code: int) -> Optional[str]:
        """仅根据 status_code 检测手类型（不验证匹配）"""
        if status_code == 0:
            return "Left"
        elif status_code == 1:
            return "Right"
        return None

    @staticmethod
    def calculate_checksum(data: bytes) -> int:
        return sum(data) & 0xFF

    @staticmethod
    def pack_data(cmd: int, payload: bytes = b'') -> bytes:
        header = struct.pack('BBB', FRAME_HEADER, cmd, len(payload))
        checksum = FrameHandler.calculate_checksum(header + payload)
        return header + payload + struct.pack('B', checksum)

    def pack_version_query(self) -> bytes:
        return self.pack_data(CommandCode.VERSION_QUERY.value)

    def pack_position_query(self) -> bytes:
        return self.pack_data(CommandCode.POSITION_QUERY.value)

    def pack_force_feedback(self) -> bytes:
        payload = struct.pack(f'{len(self._forcelist)}f', *self._forcelist)
        return self.pack_data(CommandCode.FORCE_FEEDBACK.value, payload)


# ============== 串口连接管理器 ==============

class SerialConnection:
    def __init__(self, logger: Optional['Logger'] = None, isdebug: bool = False):
        self.serial_port: Optional[serial.Serial] = None
        self.running = Event()
        self.thread: Optional[Thread] = None
        self.logger = logger
        self.isdebug = isdebug
        self._last_receive_time = time.time()
        self._last_check_time = time.time()
        self._disconnect_warned = False
        self._on_disconnect: Optional[Callable[[], None]] = None
        self._on_reconnect: Optional[Callable[[], None]] = None

    def open(self, port: str, baudrate: int) -> bool:
        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=0.001,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                exclusive=True,
            )
            self.running = Event()
            return True
        except serial.SerialException as e:
            if self.logger:
                self.logger.log('error', f"串口打开失败: {e}")
            return False

    def close(self) -> None:
        self.running.clear()
        serial_port = self.serial_port
        if serial_port and serial_port.is_open:
            try:
                cancel_read = getattr(serial_port, "cancel_read", None)
                if cancel_read:
                    cancel_read()
            except (OSError, serial.SerialException):
                pass
            try:
                serial_port.close()
            except (OSError, serial.SerialException):
                pass
        self.serial_port = None

        if self.thread and self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout=1.0)

    def start(self, data_callback: Callable[[array.array], None], 
              query_callback: Callable[[], Optional[bytes]]) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.running.set()
        self.thread = Thread(target=self._run, args=(data_callback, query_callback), daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.close()

    @staticmethod
    def _is_bad_file_descriptor_error(error: Exception) -> bool:
        if getattr(error, "errno", None) == 9:
            return True

        error_text = str(error)
        return "[Errno 9]" in error_text or "Bad file descriptor" in error_text

    def _handle_terminal_serial_error(self) -> None:
        self.running.clear()
        serial_port = self.serial_port
        self.serial_port = None

        if serial_port and getattr(serial_port, "is_open", False):
            try:
                serial_port.close()
            except (OSError, serial.SerialException):
                pass

        if not self._disconnect_warned:
            self._disconnect_warned = True
            if self._on_disconnect:
                self._on_disconnect()

    def _run(self, data_callback: Callable[[array.array], None], 
             query_callback: Callable[[], Optional[bytes]]) -> None:
        parser = FrameParser()
        sendcount = 0
        
        while self.running.is_set():
            try:
                current_time = time.time()

                # 断联检测
                if current_time - self._last_check_time >= CHECK_INTERVAL:
                    self._last_check_time = current_time
                    elapsed = current_time - self._last_receive_time

                    if elapsed > CONNECTION_TIMEOUT:
                        if not self._disconnect_warned:
                            port_name = self.serial_port.port if self.serial_port else 'unknown'
                            if self.logger:
                                self.logger.log('error', f"串口 {port_name} 超过 {CONNECTION_TIMEOUT}秒 无响应，可能已断联")
                            self._disconnect_warned = True
                            if self._on_disconnect:
                                self._on_disconnect()

                # 读取数据
                if self.serial_port and self.serial_port.in_waiting > 0:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    if data:
                        if self._disconnect_warned:
                            port_name = self.serial_port.port if self.serial_port else 'unknown'
                            if self.logger:
                                self.logger.log('info', f"串口 {port_name} 已恢复连接")
                            self._disconnect_warned = False
                            if self._on_reconnect:
                                self._on_reconnect()
                        
                        self._last_receive_time = current_time
                        for byte in data:
                            if parser.process_byte(byte):
                                if data_callback:
                                    data_callback(parser.frame_buf)
                                parser.reset()

                # 发送查询
                if query_callback:
                    sendcount += 1
                    if sendcount > QUERY_INTERVAL:
                        query_data = query_callback()
                        if query_data and self.serial_port:
                            self.serial_port.write(query_data)
                        sendcount = 0

                time.sleep(READ_INTERVAL)

            except Exception as e:
                if not self.running.is_set():
                    break
                if self._is_bad_file_descriptor_error(e):
                    self._handle_terminal_serial_error()
                    break
                if self.logger and self.isdebug:
                    self.logger.log('error', f"串口读取错误: {e}")
                time.sleep(ERROR_DELAY)

    def set_disconnect_callback(self, callback: Callable[[], None]) -> None:
        self._on_disconnect = callback

    def set_reconnect_callback(self, callback: Callable[[], None]) -> None:
        self._on_reconnect = callback


# ============== 主类：整合以上模块 ==============

class ForceSerialReader:
    def __init__(self, gettype: HandType, excludelist: Optional[List[str]] = None, 
                 baudrates: Optional[List[int]] = None, isdebug: bool = False, 
                 logger: Optional[Callable[[str, str], None]] = None):
        self.gettype = gettype
        self.isdebug = isdebug
        self.connflag = False
        self.position_frame_count = 0
        self.version: Optional[str] = None
        self.handtype: Optional[HandType] = None

        # 初始化模块
        self._logger = Logger(logger, isdebug)
        self._scanner = SerialScanner(baudrates, excludelist, self._logger)
        self._handler = FrameHandler(gettype, self._logger)
        self._connection = SerialConnection(self._logger, isdebug)
        self._serial_write_lock = threading.Lock()

        # 串口参数代理
        self.serial_port: Optional[serial.Serial] = None
        self.baudrates = self._scanner.baudrates
        self.checked_ports = self._scanner.checked_ports
        self.exclude_ports = self._scanner.exclude_ports

    # 数据属性代理
    @property
    def poslist(self) -> List[float]:
        return self._handler.poslist

    @poslist.setter
    def poslist(self, value: List[float]):
        self._handler.poslist = value

    @property
    def forcelist(self) -> List[float]:
        return self._handler.forcelist

    @forcelist.setter
    def forcelist(self, value: List[float]):
        self._handler.forcelist = value

    @property
    def realforcelist(self) -> List[int]:
        return self._handler.realforcelist

    @realforcelist.setter
    def realforcelist(self, value: List[int]):
        self._handler.realforcelist = value

    def _log(self, level: str, msg: str) -> None:
        self._logger.log(level, msg)

    # 扫描方法
    def is_usb_device(self, port_name: str) -> bool:
        return self._scanner.is_usb_device(port_name)

    def scan_serial_ports(self) -> List[str]:
        return self._scanner.scan_available_ports()

    def find_valid_ports(self, timeout: float = 2, scan_interval: float = 2) -> tuple:
        if self.isdebug:
            self._log('debug', "开始扫描串口...")
            self._log('debug', f"排除列表: {list(self.exclude_ports)}")
            self._log('debug', f"波特率组合: {self.baudrates}")

        available_ports = self.scan_serial_ports()
        if self.isdebug:
            self._log('debug', f"发现 {len(available_ports)} 个未检查的串口: {available_ports}")

        for port in available_ports:
            success, baudrate, errorcode = self.query_serial_port(port, timeout)
            
            if not success and errorcode != -2:
                if self.isdebug:
                    self._log('debug', "首次连接失败，尝试重试...")
                time.sleep(RETRY_DELAY)
                success, baudrate, errorcode = self.query_serial_port(port, timeout)

            if errorcode == -2:
                self._log('warn', f"警告: 串口 {port} 权限不足，请手动执行: sudo chmod 666 {port}")

            self.checked_ports.add(port)

            if success:
                if self.isdebug:
                    self._log('info', f"找到有效串口: {port} (波特率: {baudrate})")
                return port, baudrate, errorcode

        return None, None, None

    def _poll_serial_frames_once(self, parser: FrameParser) -> None:
        if not self.serial_port:
            return

        waiting = getattr(self.serial_port, "in_waiting", 0)
        if waiting <= 0:
            return

        data = self.serial_port.read(waiting)
        for byte in data:
            if parser.process_byte(byte):
                self._on_data_received(parser.frame_buf)
                parser.reset()

    def query_version_sync(
        self,
        retry_count: int = VERSION_QUERY_RETRY_COUNT,
        response_wait: float = WARMUP_DELAY,
    ) -> bool:
        if not self.serial_port:
            return False

        retry_count = max(1, retry_count)
        self.handtype = None
        self.version = None
        self.connflag = False
        parser = FrameParser()

        try:
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
        except Exception:
            pass

        for _ in range(retry_count):
            self.serial_port.write(self.pack_01_data())

            deadline = time.time() + response_wait
            while True:
                self._poll_serial_frames_once(parser)
                if self.handtype is not None:
                    return True
                if time.time() >= deadline:
                    break
                time.sleep(READ_INTERVAL)

        self._poll_serial_frames_once(parser)
        return self.handtype is not None

    def query_serial_port(self, port_name: str, timeout: float = 1) -> tuple:
        best_baudrate: Optional[int] = None
        errorcode: Optional[int] = None

        for baudrate in self.baudrates:
            ser: Optional[serial.Serial] = None
            try:
                ser = serial.Serial(port_name, baudrate, timeout=timeout,
                                    bytesize=serial.EIGHTBITS,
                                    parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE,
                                    exclusive=True)
                self.serial_port = ser

                if self.isdebug:
                    self._log('debug', f"串口 {port_name} 波特率 {baudrate} 预热中...")

                if self.isdebug:
                    self._log('debug', f"同步侦测串口 {port_name} 波特率 {baudrate} 是否联通...")

                detected = self.query_version_sync(response_wait=max(timeout, READ_INTERVAL))
                if ser and ser.is_open:
                    ser.close()
                self.serial_port = None

                if detected:
                    best_baudrate = baudrate
                    if self.isdebug:
                        self._log('info', f"串口 {port_name} 在 {baudrate} 波特率下有响应")
                    return True, best_baudrate, errorcode

            except serial.SerialException as e:
                # 确保清理
                if self._connection.thread:
                    self._connection.stop()
                if ser and ser.is_open:
                    ser.close()
                self.serial_port = None
                
                error_msg = str(e)
                if "No such file" in error_msg or "[Errno 2]" in error_msg:
                    errorcode = -1
                    if self.isdebug:
                        self._log('debug', f"串口设备不存在: {port_name}")
                    break
                elif "Permission denied" in error_msg or "[Errno 13]" in error_msg:
                    errorcode = -2
                    self._log('warn', f"权限被拒绝: {port_name}")
                    break
                elif "Device or resource busy" in error_msg:
                    errorcode = -3
                    if self.isdebug:
                        self._log('debug', f"设备忙: {port_name}")
                    break
                else:
                    errorcode = -99
                    if self.isdebug:
                        self._log('debug', f"串口打开失败: {e}")
                continue

        return False, None, errorcode

    # 连接方法
    def openserial(self, port: str, baudrate: int = 2000000) -> bool:
        result = self._connection.open(port, baudrate)
        if result:
            self.serial_port = self._connection.serial_port
        return result

    def start(self) -> None:
        self._connection.start(self._on_data_received, self._get_query_data)

    def stop(self) -> None:
        self._connection.stop()

    def _on_data_received(self, frame: array.array) -> None:
        self.connflag = True
        result = self._handler.handle_frame(frame)
        
        if result:
            if 'version' in result:
                self.version = result['version']
            if 'raw_handtype' in result:
                self.handtype = result['raw_handtype']
            elif 'handtype' in result:
                self.handtype = result['handtype']
            if 'poslist' in result:
                self.position_frame_count += 1
            if 'force_response' in result and self.serial_port:
                self._write_serial(self.pack_A7_data(self.forcelist))

    def _get_query_data(self) -> Optional[bytes]:
        if self.handtype is not None:
            return self.pack_03_data()
        return None

    def _write_serial(self, data: bytes) -> bool:
        if not self.serial_port:
            return False
        with self._serial_write_lock:
            self.serial_port.write(data)
        return True

    def write_force_feedback(self) -> bool:
        return self._write_serial(self.pack_04_data())

    def set_reconnect_callback(self, callback: Callable[[], None]) -> None:
        self._connection.set_reconnect_callback(callback)

    # 数据打包方法
    @staticmethod
    def calculate_checksum(data: bytes) -> int:
        return FrameHandler.calculate_checksum(data)

    def pack_01_data(self) -> bytes:
        return self._handler.pack_version_query()

    def pack_02_data(self, mastersendflag: int) -> bytes:
        payload = struct.pack('BBBBB', mastersendflag, 0, 0, 0, 0)
        return FrameHandler.pack_data(CommandCode.SET_FLAG.value, payload)

    def pack_03_data(self) -> bytes:
        return self._handler.pack_position_query()

    def pack_A3_data(self) -> bytes:
        return FrameHandler.pack_data(CommandCode.A3_POSITION.value)

    def pack_04_data(self):
        return self._handler.pack_force_feedback()

    def pack_A4_data(self, float_data):
        payload = struct.pack(f'{len(float_data)}f', *float_data)
        return FrameHandler.pack_data(CommandCode.A6_POSITION.value, payload)

    def pack_A7_data(self, float_data):
        payload = struct.pack(f'{len(float_data)}f', *float_data)
        return FrameHandler.pack_data(CommandCode.A7_FORCE.value, payload)

    # 兼容性方法
    def hex_dump(self, data):
        return ' '.join(f'{b:02X}' for b in data)

    def get_current_status(self):
        return {
            'valid_ports': [],
            'checked_ports': list(self.checked_ports),
            'exclude_ports': list(self.exclude_ports),
            'baudrates': self.baudrates
        }
