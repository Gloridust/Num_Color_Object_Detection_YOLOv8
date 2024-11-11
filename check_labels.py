import os
import config

def check_labels():
    try:
        sets = ['train', 'val']
        for subset in sets:
            labels_dir = os.path.join(config.LABELS_DIR, subset)
            images_dir = os.path.join(config.IMAGES_DIR, subset)
            image_files = [os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith('.jpg')]
            label_files = [os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt')]

            # 检查缺少标签的图片
            images_without_labels = [f for f in image_files if f not in label_files]
            if images_without_labels:
                print(f"\n{subset} 集中缺少标签文件的图片：")
                for f in images_without_labels:
                    print(f"{f}.jpg")

            # 检查标签内容
            for label_file in label_files:
                label_path = os.path.join(labels_dir, label_file + '.txt')
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line_num, line in enumerate(lines, 1):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            print(f"文件 {label_path} 第 {line_num} 行格式错误：{line.strip()}")
                            continue
                        class_id, x_center, y_center, width, height = parts
                        values = [float(x_center), float(y_center), float(width), float(height)]
                        if any(v < 0 or v > 1 for v in values):
                            print(f"文件 {label_path} 第 {line_num} 行坐标超出范围：{line.strip()}")
    except Exception as e:
        print(f"检查标签过程中出现错误: {e}")

if __name__ == "__main__":
    check_labels()