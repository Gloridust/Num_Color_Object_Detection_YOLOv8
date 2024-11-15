import socket

def send_detection_results(label, cx, cy):
    try:
        # 创建UDP套接字
        # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # server_address = ('192.168.1.100', 8080)  # 替换为实际IP和端口

        # 在检测到目标后发送数据
        message = f"{label},{cx},{cy}"
        print(f"发送数据: {message}")  # 调试时打印数据
        # sock.sendto(message.encode(), server_address)  # 若需要发送数据，取消注释
    except Exception as e:
        print(f"发送检测结果时出现错误: {e}")