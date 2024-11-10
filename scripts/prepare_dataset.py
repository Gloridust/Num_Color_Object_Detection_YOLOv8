import os
import random
import shutil
import cv2

def prepare_dataset():
    # 定义路径
    dataset_dir = './dataset'
    pics_dir = os.path.join(dataset_dir, 'pics')
    label_dir = os.path.join(dataset_dir, 'label')

    # 输出目录
    images_dir = './data/images'
    labels_dir = './data/labels'

    # 创建目录
    os.makedirs(os.path.join(images_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(images_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, 'val'), exist_ok=True)

    # 获取所有图片文件
    images = [f for f in os.listdir(pics_dir) if f.endswith('.jpg')]
    images.sort()
    random.shuffle(images)

    # 划分训练集和验证集，80%训练，20%验证
    train_ratio = 0.8
    train_count = int(len(images) * train_ratio)
    train_images = images[:train_count]
    val_images = images[train_count:]

    # 处理训练集
    for img_name in train_images:
        process_image(img_name, pics_dir, label_dir, images_dir, labels_dir, 'train')

    # 处理验证集
    for img_name in val_images:
        process_image(img_name, pics_dir, label_dir, images_dir, labels_dir, 'val')

def process_image(img_name, pics_dir, label_dir, images_dir, labels_dir, subset):
    base_name = os.path.splitext(img_name)[0]
    img_src = os.path.join(pics_dir, img_name)
    img_dst = os.path.join(images_dir, subset, img_name)
    shutil.copyfile(img_src, img_dst)

    # 处理标签
    label_src = os.path.join(label_dir, base_name + '.txt')
    label_dst = os.path.join(labels_dir, subset, base_name + '.txt')
    if os.path.exists(label_src):
        convert_label(label_src, label_dst, img_dst)
    else:
        print(f"警告：未找到图片 {img_name} 对应的标签文件。")

def convert_label(label_src, label_dst, img_path):
    """
    将自定义格式的标签转换为YOLO格式
    """
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]

    with open(label_src, 'r') as f_in, open(label_dst, 'w') as f_out:
        lines = f_in.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"标签文件 {label_src} 格式错误。")
                continue
            class_id = int(parts[0])
            x1, y1, x2, y2 = map(float, parts[1:])

            # 如果坐标是绝对像素值，需要进行归一化
            if x2 <= 1 and y2 <=1:
                # 如果坐标已经是相对值（0~1），转换为绝对像素值
                x1 *= img_width
                x2 *= img_width
                y1 *= img_height
                y2 *= img_height

            # 计算中心坐标和宽高
            x_center = (x1 + x2) / 2 / img_width
            y_center = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            # 写入YOLO格式的标签文件
            f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

if __name__ == "__main__":
    prepare_dataset() 