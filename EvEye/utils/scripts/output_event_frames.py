import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from EvEye.utils.tonic.functional.ToFrameStack import to_frame_stack_numpy
from EvEye.utils.tonic.slicers.SliceEventsAtIndices import slice_events_at_timepoints
from EvEye.utils.processor.TxtProcessor import TxtProcessor
from EvEye.utils.visualization.visualization import visualize, save_image
from tqdm import tqdm
from natsort import natsorted
time_window = 5000
sensor_size = (346, 260, 2)
root_path = Path(
    "D:/EV-Eye-main/EV_Eye_dataset/raw_data/Data_davis/user1/right/session_2_0_1/events"
)
output_base_path = Path("D:/EV-Eye-main/EV_Eye_dataset/raw_data/Data_davis/user1/right/session_2_0_1/events/outframes")
# 直接指定要读取的文件路径
data_path = root_path / "events.txt"

# 检查文件是否存在
if not data_path.exists():
    print(f"错误: 文件 '{data_path}' 不存在!")
else:
    output_base_name = data_path.stem
    output_path = output_base_path / output_base_name
    os.makedirs(output_path, exist_ok=True)
    print(f"正在处理 {output_base_name}...")

    events = TxtProcessor(data_path).load_events_from_txt()
    events_end_time = events['t'][-1]
    events_start_time = events['t'][0]
    num_frames = (events_end_time - events_start_time) // time_window
    count = 0
    for index in tqdm(range(num_frames)):
        start_time = events_start_time + index * time_window
        end_time = start_time + time_window
        event_segment = slice_events_at_timepoints(events, start_time, end_time)
        if len(event_segment) < 800:
            continue
        event_frame = to_frame_stack_numpy(
            event_segment,
            sensor_size,
            1,
            "causal_linear",
            start_time,
            end_time,
        )
        event_frame_vis = visualize(event_frame)
        event_frame_vis_name = str(output_path / f"{count:05}_{end_time}.png")
        save_image(event_frame_vis, event_frame_vis_name)
        count += 1