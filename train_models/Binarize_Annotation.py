import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def get_files_in_current_folder(folder_path="."):
    # 获取当前文件夹下所有文件的路径
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
             os.path.isfile(os.path.join(folder_path, file))]
    return files


def binarize_the_channel(image, threshold_binarization):
    """
    :param image: OpenCV类的图像
    :param threshold_binarization: 二值化的阈值
    :return: 二值化后的图像
    """
    # 将图片转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 应用阈值二值化
    ret, binary = cv2.threshold(gray, threshold_binarization, 255, cv2.THRESH_BINARY)
    return binary


def binarize_annotations(binary, min_area):
    """
    :param binary: 二值化图片路径
    :param min_area: 最小的框选面积
    :return: 框选位置
    """

    # 使用OpenCV的轮廓检测找到所有的框
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 定义阈值，用于区分较小的框和较大的框
    threshold_area = min_area

    # 循环遍历所有的轮廓
    # 创建一个新的列表来存储合并后的大框
    merged_contours = []

    # 循环遍历所有的轮廓
    for i, contour in enumerate(contours):
        # 计算轮廓的面积
        area = cv2.contourArea(contour)

        # 根据面积阈值判断是较小的框还是较大的框
        if area > threshold_area:
            x, y, w, h = cv2.boundingRect(contour)

            # 遍历已合并的大框，查找是否存在比当前框大的框
            merged = False
            for merged_contour in merged_contours:
                # 提取大框的坐标和大小
                merged_x, merged_y, merged_w, merged_h = cv2.boundingRect(merged_contour[0])

                # 判断当前框是否在合并后的大框附近
                if (
                        merged_x < x < merged_x + merged_w + 20
                        and merged_y < y < merged_y + merged_h + 20
                ):
                    # 合并当前框和合并后的大框
                    x = min(x, merged_x)
                    y = min(y, merged_y)
                    w = max(x + w, merged_x + merged_w) - x
                    h = max(y + h, merged_y + merged_h) - y

                    # 更新合并后的大框的坐标和大小
                    merged_contour[0] = np.vstack((merged_contour[0], contour))

                    merged = True
                    break

            # 如果当前框没有与合并后的大框匹配，将其添加到合并后的大框列表中
            if not merged:
                merged_contours.append([contour])
    return merged_contours


if __name__ == "__main__":
    files_path = get_files_in_current_folder("/home/yy/data/wildfire/flames2/temp")
    txt_root = "/home/yy/data/wildfire/flames2/labels/"  # labels的路径+/
    current_num = 0
    for img_path in files_path:
        # 读取图片
        txt_path = txt_root + img_path.split("/")[-1].split(".")[0] + "txt"
        current_num += 1
        image = cv2.imread(img_path)
        binary_img = binarize_the_channel(image, 120)
        merged_contours = binarize_annotations(binary_img, 300)
        # 复制图像以防止更改原始图像
        marked_img = binary_img.copy()
        # 在标注图像上画出融合后的矩形框
        for merged_contour in merged_contours:
            x, y, w, h = cv2.boundingRect(np.vstack(merged_contour))
            cv2.rectangle(marked_img, (x, y), (x + w, y + h), 255, 2)
        with open(txt_path, 'w') as f:
            # 遍历合并后的大框
            for idx, merged_contour in enumerate(merged_contours):
                x, y, w, h = cv2.boundingRect(np.vstack(merged_contour))

                # 计算中心坐标、归一化宽和归一化高
                center_x = (x + x + w) / 2 / binary_img.shape[1]
                center_y = (y + y + h) / 2 / binary_img.shape[0]
                normalized_w = w / binary_img.shape[1]
                normalized_h = h / binary_img.shape[0]

                # 将数据写入文件
                f.write(f"0 {center_x} {center_y} {normalized_w} {normalized_h}\n")

        print(f"Data written to output{current_num}")
        # 使用Matplotlib显示图像
        plt.imshow(marked_img, cmap='gray')
        plt.title('Marked Bounding Boxes without mix')
        plt.show()

    # # 创建保存二值化图片的文件夹
    # if not os.path.exists('binary_images'):
    #     os.makedirs('binary_images')
    #
    # # 保存二值化图片到文件夹
    # cv2.imwrite('binary_images/binary_image.jpg', binary)
