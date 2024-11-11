from ultralytics import YOLO
import os
import config
import torch

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def train_model():
    try:
        # 设置工作目录为项目根目录
        os.chdir(config.PROJECT_ROOT)

        # 加载YOLOv8的nano模型
        model = YOLO('yolov8n.yaml')

        # 设置设备
        device = get_device()
        print(f"使用设备: {device}")

        # 更新训练参数中的设备
        config.TRAIN_PARAMS['device'] = device

        # 开始训练
        model.train(**config.TRAIN_PARAMS)
    except Exception as e:
        print(f"训练过程中出现错误: {e}")

if __name__ == "__main__":
    train_model() 