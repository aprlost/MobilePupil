import cv2
import numpy as np
import os
from numba import jit
import time
# ====================== 参数配置 ======================
root_path = "D:/EV-Eye-main/EV_Eye_dataset/raw_data/Data_davis/user1/left/session_1_0_1/events/outframes/events"  # 输入PNG文件目录（需替换）
save_dir = "D:/EV-Eye-main/EV_Eye_dataset/raw_data/Data_davis/user1/left/session_1_0_1/events/outframes/lvboevents"  # 滤波结果保存目录（需替换）
theta_s = 1.5  # 空间邻域阈值（建议2-4）
theta_t = 0.3  # 时间累积阈值（需调试）
alpha = 0.5  # 时间窗口缩放系数（建议0.1-1.0）
epsilon = 1e-6  # 防止除零的极小值
MAX_EVENTS_PER_PIXEL = 10000
#== == == == == == == == == == == 数据加载 == == == == == == == == == == ==
# 读取并排序PNG文件
files = sorted([f for f in os.listdir(root_path) if f.endswith('.png')])
frames = []
for f in files:
    # 读取灰度图并二值化（1表示事件，0表示无事件）
    img = cv2.imread(os.path.join(root_path, f), cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    frames.append(img_bin)
frames = np.array(frames, dtype=np.uint8)  # 形状: (帧数, 高度, 宽度)
num_frames, H, W = frames.shape

# ====================== 预处理：事件时间戳 ======================
# 创建固定大小的数组存储事件时间戳和计数
# 格式: event_timestamps[y, x, i] = 第i个事件的帧索引 (-1表示无效)
#       event_counts[y, x] = 该像素的有效事件数
event_timestamps = np.full((H, W, MAX_EVENTS_PER_PIXEL), -1, dtype=np.int32)
event_counts = np.zeros((H, W), dtype=np.int32)

for t in range(num_frames):
    for y in range(H):
        for x in range(W):
            if frames[t, y, x] == 1:
                count = event_counts[y, x]
                if count < MAX_EVENTS_PER_PIXEL:
                    event_timestamps[y, x, count] = t
                    event_counts[y, x] += 1

# ====================== 计算自适应时间窗口 ======================
event_count = np.sum(frames, axis=0)  # 每个位置的总事件数
rho = event_count / num_frames  # 单位时间事件密度（帧级）
delta_T = alpha / (rho + epsilon)  # 自适应时间窗口大小

# ====================== 空间滤波核 ======================
spatial_kernel = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=np.uint8)


# ====================== Numba加速函数 ======================
@jit(nopython=True)
def process_frame_numba(current_frame, spatial_mask, event_timestamps, event_counts, delta_T, theta_t, H, W, t):
    """
    时间滤波 + 联合决策（Numba加速）
    """
    filtered = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            if spatial_mask[y, x] == 0:
                continue  # 空间滤波未通过，跳过

            # 获取当前位置在t之前的事件
            count = event_counts[y, x]
            past_events = []
            for i in range(count):
                ts = event_timestamps[y, x, i]
                if ts < t and ts != -1:
                    past_events.append(ts)

            # 计算指数衰减权重和
            time_sum = 0.0
            if len(past_events) > 0:
                for ts in past_events:
                    tau = t - ts
                    dt = delta_T[y, x]
                    time_sum += np.exp(-tau / dt)

            # 联合决策
            if current_frame[y, x] == 1 and time_sum >= theta_t:
                filtered[y, x] = 1
    return filtered


# ====================== 逐帧处理 ======================
os.makedirs(save_dir, exist_ok=True)

start_time = time.time()
for t in range(num_frames):
    # 1. 空间滤波
    current_frame = frames[t]
    neighbor_sum = cv2.filter2D(
        current_frame.astype(np.float32),
        -1,
        spatial_kernel,
        borderType=cv2.BORDER_CONSTANT
    )
    spatial_mask = (neighbor_sum >= theta_s).astype(np.uint8)

    # 2. 时间滤波 + 联合决策
    filtered_frame = process_frame_numba(
        current_frame,
        spatial_mask,
        event_timestamps,
        event_counts,
        delta_T,
        theta_t,
        H, W, t
    )

    # 3. 保存结果
    save_path = os.path.join(save_dir, f"filtered_{t:04d}.png")
    cv2.imwrite(save_path, filtered_frame * 255)

    # 进度显示
    if (t + 1) % 10 == 0:
        elapsed = time.time() - start_time
        eta = elapsed * (num_frames - (t + 1)) / (t + 1)
        print(f"已处理 {t + 1}/{num_frames} 帧 | 用时: {elapsed:.2f}s | 预计剩余: {eta:.2f}s")

print(f"✅ 处理完成！结果已保存至：{save_dir} | 总用时: {time.time() - start_time:.2f}s")