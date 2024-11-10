from ultralytics import YOLO

def train_model():
    # 加载YOLOv8的nano模型
    model = YOLO('yolov8n.yaml')

    # 开始训练
    model.train(
        data='data.yaml',  # 数据集配置文件
        epochs=50,
        imgsz=640,
        batch=16,
        device=0  # 使用GPU，如无GPU可设置为'cpu'
    )

if __name__ == "__main__":
    train_model() 