import cv2
import os

# 创建保存帧的文件夹
if not os.path.exists('/home/yy/data/wildfire/flames2/frame2_images'):
    os.makedirs('/home/yy/data/wildfire/flames2/flame2_images')
# 读取视频
for i in range(2):
    video = cv2.VideoCapture(f'/home/yy/data/wildfire/flames2/Flame2_videos/#{i+1}) IR Video {i+1}.MP4')
    IR_num = i + 6

    # 循环读取视频中的每一帧
    frame_count = 0
    while True:
        # 读取一帧
        ret, frame = video.read()

        # 如果读取失败，退出循环
        if not ret:
            break

        # 每隔10帧保存一次
        if frame_count % 10 == 0:
            # 保存帧到文件夹
            cv2.imwrite(f'/home/yy/data/wildfire/flames2/temp/flame_IR_{IR_num}_{frame_count // 10}.jpg', frame)

        # 帧数加1
        frame_count += 1

    # 释放视频
    video.release()
