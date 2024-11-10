from ultralytics import YOLO
import os

def train_model():
    # 设置工作目录为项目根目录
    os.chdir('/home/gloridust/Documents/Github/Num_Color_Object_Detection_YOLOv8')

    # 加载YOLOv8的nano模型
    model = YOLO('yolov8n.yaml')

    # 开始训练
    model.train(
        data='/home/gloridust/Documents/Github/Num_Color_Object_Detection_YOLOv8/data.yaml',  # 使用绝对路径
        epochs=50,
        imgsz=640,
        batch=16,
        device=0  # 使用GPU，如无GPU可设置为'cpu'
    )

if __name__ == "__main__":
    train_model() 