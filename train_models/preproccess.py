import os
import random
import shutil

if __name__ == "__main__":
    train_images_folder = "/home/yy/data/wildfire/D-Fire-001/train/images"
    train_labels_folder = "/home/yy/data/wildfire/D-Fire-001/train/labels"
    val_images_folder = "/home/yy/data/wildfire/D-Fire-001/val/images"
    val_labels_folder = "/home/yy/data/wildfire/D-Fire-001/val/labels"

    # 如果验证文件夹不存在，则创建它们
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)

    # 获取列表集中的映像文件列表
    image_files = os.listdir(train_images_folder)

    # 计算要移动到验证集的图像数量
    num_val_images = int(0.1 * len(image_files))

    # 随机选择要移动的图像
    val_image_files = random.sample(image_files, num_val_images)

    # 将选定的图像及其相应的标签移动到验证集中
    for image_file in val_image_files:
        # 移动图像文件
        image_src = os.path.join(train_images_folder, image_file)
        image_dst = os.path.join(val_images_folder, image_file)
        shutil.move(image_src, image_dst)

        # 移动标签文件
        label_file = image_file.replace(".jpg", ".txt")
        label_src = os.path.join(train_labels_folder, label_file)
        label_dst = os.path.join(val_labels_folder, label_file)
        shutil.move(label_src, label_dst)
    print("OK")
