import cv2
from ultralytics import YOLO
from utils import send_detection_results
import config
import torch

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def detect():
    try:
        # 加载训练好的模型
        model = YOLO(config.MODEL_PATH)  # 使用配置中的模型路径

        # 设置设备
        device = get_device()
        print(f"使用设备: {device}")

        # 打开摄像头
        cap = cv2.VideoCapture(0)  # 根据实际摄像头调整设备ID

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 推理
            results = model(frame, device=device)

            # 解析结果
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # 获取坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # 计算中心坐标
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # 获取类别和置信度
                    cls_id = int(box.cls[0])
                    conf = box.conf[0]
                    label = model.names[cls_id]

                    # 绘制矩形和中心点
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # 输出中心坐标
                    print(f"检测到{label}，中心坐标：({cx}, {cy})")

                    # 发送检测结果
                    send_detection_results(label, cx, cy)

            # 显示结果
            cv2.imshow('YOLOv8 Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"检测过程中出现错误: {e}")
    finally:
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect() 