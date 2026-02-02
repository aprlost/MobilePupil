import os
from PIL import Image
import numpy as np
from NumpyEventFrameRandomAffine import NumpyEventFrameRandomAffine

def augment_and_save_images(root_path, save_dir):
    # 初始化增广器
    augmenter = NumpyEventFrameRandomAffine(size=(260, 346))

    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 遍历根目录下的所有文件
    for filename in os.listdir(root_path):
        if filename.endswith('.png'):
            # 读取PNG文件
            file_path = os.path.join(root_path, filename)
            image = Image.open(file_path)
            event_frame = np.array(image)

            # 调整事件帧的维度以适应增广器的输入要求
            if event_frame.ndim == 2:  # 单通道图像
                event_frame = np.expand_dims(event_frame, axis=0)
                event_frame = np.expand_dims(event_frame, axis=0)
            elif event_frame.ndim == 3:  # 多通道图像
                event_frame = np.transpose(event_frame, (2, 0, 1))
                event_frame = np.expand_dims(event_frame, axis=0)

            # 进行增广操作
            augmented_event_frame = augmenter.transform_event_frame(event_frame)

            # 处理增广后的事件帧，将其转换为合适的图像格式
            if augmented_event_frame.ndim == 3:
                augmented_event_frame = np.squeeze(augmented_event_frame, axis=0)
            elif augmented_event_frame.ndim == 4:
                augmented_event_frame = np.squeeze(augmented_event_frame, axis=0)
                augmented_event_frame = np.transpose(augmented_event_frame, (1, 2, 0))

            augmented_event_frame = augmented_event_frame.astype(np.uint8)

            # 创建增广后图像的保存路径
            save_path = os.path.join(save_dir, f'aug_{filename}')

            # 保存增广后的图像
            augmented_image = Image.fromarray(augmented_event_frame)
            augmented_image.save(save_path)

            print(f'Saved augmented image: {save_path}')
if __name__ == "__main__":
    root_path = 'D:/EV-Eye-main/EV_Eye_dataset/raw_data/Data_davis/user1/left/session_1_0_1/events/outframes/events'  # 替换为实际的根目录
    save_dir = 'D:/EV-Eye-main/EV_Eye_dataset/raw_data/Data_davis/user1/left/session_1_0_1/events/outframes/exevents'    # 替换为实际的保存目录
    augment_and_save_images(root_path, save_dir)