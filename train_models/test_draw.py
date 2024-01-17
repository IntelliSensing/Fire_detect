from PIL import Image, ImageDraw

def draw_rectangles(image_path, output_path, rectangles):
    # 打开图像
    image = Image.open(image_path)

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
        draw.rectangle([x, y, x_right, y_bottom], outline="green", width=2)
        # 添加标签信息
        label = "smoke" if category == 0 else "fire"
        draw.text((x, y), label, fill="red")
    # 保存绘制好矩形框的图像
    image.save(output_path)

if __name__ == "__main__":
    # 指定图像路径、输出路径和矩形框信息的txt文件路径
    image_path = "/home/yy/data/wildfire/WEB10974.jpg"
    output_path = "/home/yy/data/wildfire/output_image.jpg"
    txt_file_path = "/home/yy/data/wildfire/web.txt"

    # 从txt文件中读取矩形框信息
    with open("/home/yy/data/wildfire/test123.txt", 'r') as file:
        # 假设每行包含五个整数，第一个整数为类别（0或1），后面四个整数为矩形的左上角坐标和宽高
        rectangles = [list(map(float, line.strip().split())) for line in file]
    if not rectangles:
        print("rectangles")
    # 在图像中绘制矩形框并保存
    # draw_rectangles(image_path, output_path, rectangles)
