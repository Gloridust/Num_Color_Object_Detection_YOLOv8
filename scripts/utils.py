import socket

def send_detection_results(label, cx, cy):
    # 创建UDP套接字
    # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # server_address = ('192.168.1.100', 8080)  # 替换为实际IP和端口

    # 在检测到目标后发送数据
    message = f"{label},{cx},{cy}"
    print(f"发送数据: {message}")  # 在调试阶段打印数据
    # sock.sendto(message.encode(), server_address) 