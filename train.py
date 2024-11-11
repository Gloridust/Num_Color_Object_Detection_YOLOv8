from ultralytics import YOLO
import os
import config
import torch
import logging

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

        # 开始训练
        results = model.train(**config.TRAIN_PARAMS)

        # 记录每个epoch的损失和F1分数
        best_f1 = 0
        patience_counter = 0
        patience = config.EARLY_STOPPING['patience']

        for epoch, metrics in enumerate(results):
            loss = metrics['loss']
            f1_score = metrics['f1']
            logging.info(f"Epoch {epoch + 1}: Loss = {loss:.4f}, F1 Score = {f1_score:.4f}")

            # 早停机制
            if f1_score > best_f1:
                best_f1 = f1_score
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logging.info(f"早停触发于Epoch {epoch + 1}")
                print(f"早停触发于Epoch {epoch + 1}")
                break

    except Exception as e:
        logging.error(f"训练过程中出现错误: {e}")
        print(f"训练过程中出现错误: {e}")

if __name__ == "__main__":
    train_model() 