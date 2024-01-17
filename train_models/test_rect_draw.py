import torch
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image, ImageDraw
import os

test_color_list = ["black", "blue"]
right_color_list = ["green", "pink"]


def get_files_in_current_folder(folder_path="."):
    # 获取当前文件夹下所有文件的路径
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
             os.path.isfile(os.path.join(folder_path, file))]
    return files


# 定义预处理函数
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # 输入大小
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def draw_rectangles(image, output_path, rectangles):
    # 获取图像的宽和高
    image_width, image_height = image.size

    # 创建一个可在图像上绘制的对象
    draw = ImageDraw.Draw(image)

    # 遍历矩形框列表并绘制每个矩形
    for rect in rectangles:
        category, norm_center_x, norm_center_y, norm_width, norm_height = map(float, rect)

        # 计算矩形框在图像中的实际坐标和尺寸
        center_x = int(norm_center_x * image_width)
        center_y = int(norm_center_y * image_height)
        width = int(norm_width * image_width)
        height = int(norm_height * image_height)

        # 计算矩形框的左上角和右下角坐标
        x = center_x - width // 2
        y = center_y - height // 2
        x_right = center_x + width // 2
        y_bottom = center_y + height // 2

        # 绘制矩形框
        draw.rectangle([x, y, x_right, y_bottom], outline=right_color_list[int(category)], width=2)
        # 添加标签信息
        label = "smoke" if category == 0 else "fire"
        draw.text((x, y), label, fill=right_color_list[int(category)], font_size=1)
    # 保存绘制好矩形框的图像
    image.save(output_path)


if __name__ == "__main__":
    # 加载已经训练好的 YOLO 模型
    model = YOLO('/home/yy/data/wildfire/train_models/wildfire/test00113/weights/best.pt')  # load an official model
    # model.load_state_dict(torch.load('/home/yy/data/wildfire/train_models/wildfire/test00113/weights/best.pt'))
    # model.eval()

    folder_path = "/home/yy/data/wildfire/D-Fire-001/temp/"  # "/home/yy/data/wildfire/D-Fire-001/test/"
    # 预处理输入图像
    image_path = get_files_in_current_folder(folder_path + "images")
    # image_path = get_files_in_current_folder(folder_path + "Fire")
    txt_path = folder_path + "labels/"
    output_path = "/home/yy/data/wildfire/D-Fire-001/temp_output/"  # "/home/yy/data/wildfire/test_output/"
    confidence_threshold = 0.2  # 置信度阈值设置
    for num in range(len(image_path)):
        rectangles = []
        input_image = preprocess_image(image_path[num])
        # 进行推理
        with torch.no_grad():
            detections = model(input_image)

        # 解析检测结果
        # 请根据您的模型和数据结构调整以下代码
        # 通常，YOLO 模型的输出包含边界框的坐标、类别置信度和类别标签
        # print(detections[0].boxes)
        # 您提供的数据
        result_data = {
            "boxes": detections[0].boxes,  # 替换为实际的边界框坐标
            "orig_img": detections[0].orig_img,  # 替换为实际的原始图像像素值
            "names": {0: "{0: 'smoke'}", 1: "{1: 'fire'}"},
            "class": detections[0].boxes.cls,
            "confidence": detections[0].boxes.conf
        }

        # 获取原始图像的数据
        orig_img = result_data["orig_img"][0].numpy().transpose((1, 2, 0))  # 调整通道顺序

        # 创建图像的副本，以免修改原始图像
        image_with_boxes = Image.fromarray((orig_img * 255).astype('uint8'))

        # 获取绘图对象
        draw = ImageDraw.Draw(image_with_boxes)
        temp_num = 0
        for box in result_data["boxes"].xywh:
            # print(box[0])
            center_x, center_y, width, height = box
            # 计算矩形框的左上角和右下角坐标
            x = center_x - width // 2
            y = center_y - height // 2
            x_right = center_x + width // 2
            y_bottom = center_y + height // 2
            color = test_color_list[int(result_data["class"][temp_num])]
            confidence = result_data["confidence"][temp_num]
            if confidence >= confidence_threshold:
                # 绘制矩形框
                draw.rectangle([x, y, x_right, y_bottom], outline=color, width=2)
                text = result_data["names"][int(result_data["class"][temp_num])] + f" {confidence:.2f}"
                draw.text((x, y), text, fill=color, font_size=1)
                # 显示带有边界框的图像
                temp_num += 1
            else:
                continue
        name = image_path[num].split("/")[-1]
        image_with_boxes.save(output_path + name)
        txt_name = txt_path + name.split(".")[0] + ".txt"
        with open(txt_name, 'r') as file:
            # 假设每行包含五个整数，第一个整数为类别（0或1），后面四个整数为矩形的左上角坐标和宽高
            rectangles = [list(map(float, line.strip().split())) for line in file]
        # print(txt_name, end=" ")
        # print(image_path[num])
        if not rectangles:
            continue
        draw_rectangles(image_with_boxes, output_path + name, rectangles)
        # 可以使用可视化库（如matplotlib）来可视化检测结果

