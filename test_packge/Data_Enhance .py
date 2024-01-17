import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2 as cv
from torchvision.transforms import ToPILImage
import numpy as np
import os


def get_files_in_current_folder(folder_path="."):
    # 获取当前文件夹下所有文件的路径
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
             os.path.isfile(os.path.join(folder_path, file))]
    return files


def get_train_transform():
    return A.Compose([
        A.RandomResizedCrop(height=640, width=640, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.5),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))


def get_rect_txt(txt_path):
    txt_name = txt_path
    with open(txt_name, 'r') as file:
        # 假设每行包含五个整数，第一个整数为类别（0或1），后面四个整数为矩形的左上角坐标和宽高
        rectangles = [list(map(float, line.strip().split())) for line in file]
    return rectangles


def get_real_coord(img, normalized_coordinates, xywh=True):
    image_height, image_width = img.shape[1:]
    # print(img.shape)
    # print(f"{image_height},{image_width}")
    real_rect = []
    for rect in normalized_coordinates:
        norm_center_x, norm_center_y, norm_width, norm_height = map(float, rect)
        # print(f"{norm_center_x},{norm_center_y},{norm_width},{norm_height}")
        # 计算矩形框在图像中的实际坐标和尺寸
        center_x = int(norm_center_x * image_width)
        center_y = int(norm_center_y * image_height)
        width = int(norm_width * image_width)
        height = int(norm_height * image_height)
        if xywh:
            # 计算矩形框的左上角和右下角坐标
            x = center_x - width // 2
            y = center_y - height // 2
            x_right = center_x + width // 2
            y_bottom = center_y + height // 2
            real_rect.append([x, y, x_right, y_bottom])
        else:
            real_rect.append([center_x, center_y, width, height])
    return real_rect


def get_category_rect(rect_txt):
    category = []
    rect = []
    for temp in rect_txt:
        category.append(int(temp[0]))
        rect.append(list(temp[1:]))
    return category, rect


def data_enhance(img_path, box_path, img_storage_path, box_storage_path):
    transform = get_train_transform()
    image = cv.imread(img_path)
    re_txt = get_rect_txt(box_path)
    category_ids, bboxes = get_category_rect(re_txt)
    try:
        augmented = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    except:
        return 1
    augmented_image = augmented['image']  # 数据增强后的图片
    augmented_bboxes = augmented['bboxes']  # 数据增强后的标注框，保存为归一化中心点坐标+宽+高
    # rect_box = get_real_coord(augmented_image, augmented_bboxes)
    # 将 PyTorch 张量转换为 PIL 图像
    to_pil = ToPILImage()
    pil_image = to_pil(augmented_image)
    image_np = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
    # for box in rect_box:
    #     image_np = cv.rectangle(image_np, box[0: 2], box[2:], thickness=5, color=(255, 180, 10))
    # 保存图像为 JPG 文件
    cv.imwrite(img_storage_path, image_np)
    with open(box_storage_path, 'w') as f:
        for i in range(len(augmented_bboxes)):
            norm_center_x, norm_center_y, norm_width, norm_height = map(float, augmented_bboxes[i])
            f.write(f"{category_ids[i]} {norm_center_x} {norm_center_y} {norm_width} {norm_height}\n")
    return 0


if __name__ == "__main__":
    img_files = get_files_in_current_folder("/home/yy/data/wildfire/D-Fire-001/train/images")
    box_root_path = "/home/yy/data/wildfire/D-Fire-001/train/labels/"
    img_stor_root = "/home/yy/data/wildfire/data_enhance/images/"
    box_stor_root = "/home/yy/data/wildfire/data_enhance/labels/"
    # box_files = get_files_in_current_folder("/home/yy/data/wildfire/D-Fire-001/train/labels")
    num = 0
    for img_file in img_files:
        img_path_ = img_file
        box_path_ = box_root_path + img_file.split("/")[-1].split(".")[0] + ".txt"
        img_stor_path = img_stor_root + "data_enhance_" + img_path_.split('/')[-1]
        box_stor_path = box_stor_root + "data_enhance_" + box_path_.split('/')[-1]
        temp = data_enhance(img_path_, box_path_, img_stor_path, box_stor_path)
        num += 1 - temp
        print(f"第{num}次数据增强")
    print("all over")
