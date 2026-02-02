import cv2
import numpy as np
from pathlib import Path
from EvEye.utils.visualization.visualization import *
from natsort import natsorted
import matplotlib.pyplot as plt

# 全局设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def get_masks(root_path, suffixes=['.png', '.PNG'], require_mask_suffix=False):
    """获取指定目录下的掩码图像文件"""
    if not root_path.exists():
        print(f"错误: 目录 '{root_path}' 不存在")
        return []
    if not root_path.is_dir():
        print(f"错误: '{root_path}' 不是一个目录")
        return []

    print(f"正在扫描目录: {root_path}")
    all_png_files = [file for file in root_path.iterdir()
                     if file.suffix.lower() in [s.lower() for s in suffixes]]
    print(f"找到 {len(all_png_files)} 个PNG文件")

    if require_mask_suffix:
        masks = [file for file in all_png_files
                 if file.stem.split("_")[-1] == "mask"]
        print(f"其中 {len(masks)} 个文件符合'mask'命名规则")
    else:
        masks = all_png_files
        print("所有PNG文件都被视为掩码文件")

    if masks:
        print("找到的掩码文件:")
        for i, file in enumerate(masks[:5]):
            print(f"  {i + 1}. {file.name}")
        if len(masks) > 5:
            print(f"  ... 等 {len(masks)} 个文件")
    else:
        print("未找到符合条件的掩码文件")
    return natsorted(masks)


def extract_ellipse_data(contour):
    """从轮廓中提取椭圆六元组数据 (t, x, y, a, b, theta)"""
    if len(contour) > 5:
        ellipse = cv2.fitEllipse(contour)
        center = ellipse[0]
        axes = ellipse[1]
        angle = ellipse[2]

        # 确保 a >= b
        a, b = sorted(axes, reverse=True)

        # 确保 theta 在 [0°, 180°) 范围内
        theta = angle % 180

        # 构造六元组: (索引, 中心x, 中心y, 长轴, 短轴, 旋转角度)
        t = 0  # 此处t可根据需求自定义，示例中先设为0
        x = center[0]
        y = center[1]

        return (t, x, y, a, b, theta)
    return None


def save_ellipse_data_to_txt(ellipse_data_list, output_file):
    """将椭圆数据保存到txt文件"""
    with open(output_file, 'w') as f:
        for data in ellipse_data_list:
            if data:
                # 格式: t x y a b theta
                line = f"{data[0]:.0f} {data[1]:.2f} {data[2]:.2f} {data[3]:.2f} {data[4]:.2f} {data[5]:.2f}\n"
                f.write(line)
    print(f"椭圆数据已保存到 {output_file}")


# 配置路径
root_path = Path("D:/EV-Eye-main/Data_davis_predict/user1/right/session1_0_2/frames")
output_file = "D:/EV-Eye-main/EV_Eye_dataset/raw_data/Data_davis/user1/right/session_1_0_2/events/ellipse_data.txt"

# 获取所有掩码
masks = get_masks(root_path)
if not masks:
    print("未找到掩码文件，程序退出")
    exit()

print(f"共找到 {len(masks)} 个掩码文件，开始处理...")

all_ellipse_data = []

# 遍历每个掩码文件
for i, mask_path in enumerate(masks):
    print(f"处理第 {i + 1}/{len(masks)} 个掩码: {mask_path.name}")

    # 提取文件名中两个下划线之间的数字作为t
    parts = mask_path.stem.split("_")
    if len(parts) >= 3:
        try:
            t = int(parts[1])
        except ValueError:
            print(f"文件名 {mask_path.name} 中两个下划线间不是有效数字，跳过此文件")
            continue
    else:
        print(f"文件名 {mask_path.name} 格式不正确，跳过此文件")
        continue

    # 加载图像
    raw_image = load_image(str(mask_path))[0]
    gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

    # 查找轮廓
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 处理每个轮廓
    for contour in contours:
        ellipse_data = extract_ellipse_data(contour)
        if ellipse_data:
            # 保存椭圆数据，t设为文件名中提取的数字
            all_ellipse_data.append((t, ellipse_data[1], ellipse_data[2],
                                     ellipse_data[3], ellipse_data[4], ellipse_data[5]))

# 保存所有椭圆数据到txt文件
save_ellipse_data_to_txt(all_ellipse_data, output_file)
'''
# 可视化最后一个掩码的处理结果（可选）
if masks:
    last_mask = masks[-1]
    raw_image = load_image(str(last_mask))[0]
    gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    canvas = np.zeros_like(raw_image)
    for contour in contours:
        if len(contour) > 5:
            ellipse = cv2.fitEllipse(contour)
            draw_ellipse(canvas, ellipse)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(raw_image)
    plt.title("原始图像")
    plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.imshow(canvas)
    plt.title("椭圆拟合结果")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
'''