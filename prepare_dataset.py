import os
import random
import shutil
import config

def prepare_dataset():
    try:
        # 使用配置中的路径
        pics_dir = config.PICS_DIR
        label_dir = config.LABEL_DIR
        images_dir = config.IMAGES_DIR
        labels_dir = config.LABELS_DIR

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
    except Exception as e:
        print(f"数据集准备过程中出现错误: {e}")

def process_image(img_name, pics_dir, label_dir, images_dir, labels_dir, subset):
    try:
        base_name = os.path.splitext(img_name)[0]
        img_src = os.path.join(pics_dir, img_name)
        img_dst = os.path.join(images_dir, subset, img_name)
        shutil.copyfile(img_src, img_dst)

        # 复制标签
        label_src = os.path.join(label_dir, base_name + '.txt')
        label_dst = os.path.join(labels_dir, subset, base_name + '.txt')
        if os.path.exists(label_src):
            shutil.copyfile(label_src, label_dst)
        else:
            print(f"警告：未找到图片 {img_name} 对应的标签文件 {label_src}。")
    except Exception as e:
        print(f"处理图片 {img_name} 时出现错误: {e}")

if __name__ == "__main__":
    prepare_dataset() 