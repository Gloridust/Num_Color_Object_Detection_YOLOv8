import os
import random
import shutil

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

    # 复制标签
    label_src = os.path.join(label_dir, base_name + '.txt')
    label_dst = os.path.join(labels_dir, subset, base_name + '.txt')
    if os.path.exists(label_src):
        shutil.copyfile(label_src, label_dst)
    else:
        print(f"警告：未找到图片 {img_name} 对应的标签文件 {label_src}。")

if __name__ == "__main__":
    prepare_dataset() 