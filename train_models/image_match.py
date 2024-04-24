import cv2
import numpy as np
from test_packge.test_rect_draw import get_files_in_current_folder


def img_capture(init_img, target_img, target_box):
    # Create TrackerKCF instance
    tracker = cv2.TrackerKCF_create()
    # Initialize tracker
    tracker.init(init_img, target_box)
    # Update tracker
    success, bbox = tracker.update(target_img)
    # box = [int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2), bbox[2], bbox[3]]
    return bbox


def cv_draw_rect(image, boxs, save_path, color=(138, 130, 238)):
    img = image
    for box in boxs:
        center_x, center_y, width, height = map(float, box)
        top_left = (int(center_x - width / 2), int(center_y - height / 2))
        bottom_right = (int(center_x + width / 2), int(center_y + height / 2))
        # 绘制矩形
        cv2.rectangle(img, top_left, bottom_right, color, 4)
        # 保存图片
    cv2.imwrite(save_path, img)


def calculate_iou(box1, box2):
    """
    Calculate the IoU (Intersection over Union) of two target boxes
    :param box1:[x, y, width, height]
    :param box2:[x, y, width, height]
    :return:IoU
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 计算相交矩形的左上角坐标和右下角坐标
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0  # 没有相交的部分，IoU为0

    # 计算相交矩形的面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算两个目标框的面积
    box1_area = w1 * h1
    box2_area = w2 * h2

    # 计算IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou


# 定义一个函数来匹配目标框
def match_boxes(boxes1, boxes2, threshold=0.5):
    """
    Match target box in two frames
     boxes1 and boxes2 are lists of target boxes respectively, and each target box is in the form of [x, y, width, height]
     threshold is the IoU threshold, used to determine whether two target boxes match.
     Returns a list containing target boxes that are annotated in boxes1 but not in boxes2
    """
    unmatched_boxes = []

    for box1 in boxes1:
        matched = False
        for box2 in boxes2:
            iou = calculate_iou(box1, box2)  # 忽略类别信息，只计算位置和尺寸的IoU
            if iou >= threshold:
                matched = True
                break
        if not matched:
            unmatched_boxes.append(box1)

    return unmatched_boxes


if __name__ == "__main__":
    image_folder = "/home/yy/data/wildfire/temp_IRimgs/images"
    images = get_files_in_current_folder(image_folder)
    txt_path = "/home/yy/data/wildfire/temp_IRimgs/labels"
    save_path = '/home/yy/data/wildfire/temp_IRimgs/save_/'
    # txts = get_files_in_current_folder(txt_path)
    for i in range(len(images)):
        if i+1 < len(images):
            frame1 = cv2.imread(images[i])
            frame2 = cv2.imread(images[i+1])
        else:
            break
        name1 = images[i].split('/')[-1].split('.')[0]
        name2 = images[i+1].split('/')[-1].split('.')[0]
        with open(txt_path + "/" + name1 + ".txt", mode='r') as f:
            bbox1 = [list(map(float, line.strip().split())) for line in f]
        with open(txt_path + "/" + name2 + ".txt", mode='r') as f:
            bbox2 = [list(map(float, line.strip().split())) for line in f]

        real_boxs = []
        matched_bbox = []
        for box in bbox1:
            norm_center_x, norm_center_y, norm_width, norm_height = map(float, box[1:])
            image_width = 640
            image_height = 512
            center_x = int(norm_center_x * image_width)
            center_y = int(norm_center_y * image_height)
            width = int(norm_width * image_width)
            height = int(norm_height * image_height)
            real_box = [center_x, center_y, width, height]
            temp = img_capture(frame1, frame2, real_box)
            matched_bbox.append(temp)
        # cv_draw_rect(frame2, matched_bbox, save_path + name2 + "match_new" + str(i) + ".jpg", color=(123, 56, 148))
        # cv_draw_rect(frame1, real_boxs, save_path + name1 + "f1_new" + str(i) + ".jpg")
            # real_box = [center_x-int(width/2), center_y-int(height/2), width, height]
            real_box = [center_x, center_y, width, height]
            real_boxs.append([center_x, center_y, width, height])
            temp = img_capture(frame1, frame2, real_box)
            if temp is not None:
                matched_bbox.append(temp)
        real_boxs2 = []
        for box2 in bbox2:
            norm_center_x, norm_center_y, norm_width, norm_height = map(float, box2[1:])
            image_width = 640
            image_height = 512
            center_x = int(norm_center_x * image_width)
            center_y = int(norm_center_y * image_height)
            width = int(norm_width * image_width)
            height = int(norm_height * image_height)
            real_box = [center_x, center_y, width, height]
            real_boxs2.append(real_box)
        if matched_bbox:
            # unmatched = match_boxes(matched_bbox, real_boxs2)
            cv_draw_rect(frame2, matched_bbox, save_path + name2+"match_"+str(i)+".jpg", color=(123, 56, 148))
            cv_draw_rect(frame2, real_boxs2, save_path + name2+"_"+str(i)+".jpg")
            cv_draw_rect(frame1, real_boxs, save_path + name1 + "f1_" + str(i) + ".jpg")
        else:
            print("nothing")
