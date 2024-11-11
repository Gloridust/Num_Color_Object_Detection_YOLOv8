from ultralytics import YOLO
import os
import config
import torch
import logging
from callbacks import MetricsLogger

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def setup_logging():
    logging.basicConfig(
        filename='train.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def train_model():
    setup_logging()
    try:
        # 设置工作目录为项目根目录
        os.chdir(config.PROJECT_ROOT)

        # 加载YOLOv8的nano模型
        model = YOLO('yolov8n.yaml')

        # 设置设备
        device = get_device()
        logging.info(f"使用设备: {device}")
        print(f"使用设备: {device}")

        # 更新训练参数中的设备
        config.TRAIN_PARAMS['device'] = device

        # 创建自定义回调函数实例
        metrics_logger = MetricsLogger(patience=config.EARLY_STOPPING['patience'])

        # 开始训练
        model.train(**config.TRAIN_PARAMS, callbacks=[metrics_logger])

    except Exception as e:
        logging.error(f"训练过程中出现错误: {e}")
        print(f"训练过程中出现错误: {e}")

if __name__ == "__main__":
    train_model() 