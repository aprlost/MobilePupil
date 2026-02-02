
import os
import cv2
import random
import torch
import tonic
import math
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from natsort import natsorted
from pathlib import Path

from EvEye.utils.tonic.functional.ToFrameStack import to_frame_stack_numpy
from EvEye.utils.tonic.functional.CutMaxCount import cut_max_count
from EvEye.dataset.DavisEyeEllipse.utils import *

DOWN_RATIO = 4


def load_memmap(data_file, info_file):
    with open(info_file, "r") as f:
        lines = f.readlines()
        shape_line = lines[0].strip()
        dtype_line = lines[1].strip()
        shape_str = shape_line.split(": ")[1]
        shape = tuple(int(num) for num in shape_str.strip("()").split(",") if num.strip())
        dtype_str = dtype_line.split(": ")[1]
        dtype = eval(dtype_str)
    return np.memmap(data_file, dtype=dtype, mode="r", shape=shape)


def load_cached_structed_ellipses(ellipses_path):
    ellipses_list = []
    ellipses_path = Path(ellipses_path)
    ellipses_paths = natsorted(ellipses_path.glob("ellipses_batch_*.memmap"))
    ellipses_info_paths = natsorted(ellipses_path.glob("ellipses_batch_info_*.txt"))
    for ellipses_p, ellipses_info_p in zip(ellipses_paths, ellipses_info_paths):
        ellipses_list.append(load_memmap(ellipses_p, ellipses_info_p))
    return ellipses_list


def load_event_segment(index, data_base_path, batch_size):
    data_base_path = Path(data_base_path)
    batch_id = index // batch_size
    event_id_in_batch = index % batch_size
    events_batch_path = data_base_path / f"events_batch_{batch_id}.memmap"
    events_info_path = data_base_path / f"events_batch_info_{batch_id}.txt"
    event_indices_path = data_base_path / f"events_indices_{batch_id}.npy"
    events = load_memmap(events_batch_path, events_info_path)
    indices = np.load(event_indices_path)
    start_index, end_index = indices[event_id_in_batch]
    return events[start_index:end_index]


def load_ellipse(index, ellipse_base_path):
    ellipse_base_path = Path(ellipse_base_path)

    ellipses_path = ellipse_base_path / f"ellipses_batch_0.memmap"
    ellipses_info_path = ellipse_base_path / f"ellipses_batch_info_0.txt"
    ellipses = load_memmap(ellipses_path, ellipses_info_path)
    return ellipses[index]



class DavisEyeEllipseDataset(Dataset):
    def __init__(
            self,
            root_path: Path | str,
            split="train",
            caching_events_batch_size=5000,
            sensor_size=(346, 260, 2),
            events_interpolation="causal_linear",
            pupil_area=200,
            num_classes=1,
            default_resolution=(256, 256),
            **kwargs,
    ):
        super(DavisEyeEllipseDataset, self).__init__()
        self.root_path = Path(root_path)
        self.split = split
        self.caching_events_batch_size = caching_events_batch_size

        self.data_path_left = self.root_path / self.split / "cached_data_left"
        self.ellipse_path_left = self.root_path / self.split / "cached_ellipse_left"
        self.data_path_right = self.root_path / self.split / "cached_data_right"
        self.ellipse_path_right = self.root_path / self.split / "cached_ellipse_right"

        self.sensor_size = sensor_size
        self.events_interpolation = events_interpolation
        self.pupil_area = pupil_area
        self.num_classes = num_classes
        self.max_objs = 1
        self.default_resolution = default_resolution

        self.total_frames = self.get_nums()
        print(
            f"Dataset for '{self.split}' split initialized (Integral Pose Mode). Total frames: {self.total_frames}")

    def get_nums(self):

        num_frames_list = []
        try:
            ellipses_list = load_cached_structed_ellipses(self.ellipse_path_left)
            for ellipses in ellipses_list:
                num_frames_list.append(len(ellipses))
            return sum(num_frames_list)
        except FileNotFoundError:
            print(f"Warning: Ellipse path not found for '{self.split}/left'. Returning 0.")
            return 0

    def get_transforms(self):

        if self.split == "train":
            return A.ReplayCompose([
                A.Resize(self.default_resolution[0], self.default_resolution[1]),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15,
                                   interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                                   p=1.0, value=0),

            ], additional_targets={'image0': 'image'})

        else:  # split == "val"
            return A.ReplayCompose([
                A.Resize(self.default_resolution[0], self.default_resolution[1])
            ], additional_targets={'image0': 'image'})



    def get_event_transforms(self):

        if self.split == "train":
            return tonic.transforms.Compose([
                tonic.transforms.DropEvent(p=0.3),
                tonic.transforms.DropEventByArea(sensor_size=self.sensor_size, area_ratio=0.1)
            ])
        else:
            return tonic.transforms.Compose([])

    def transform_ellipse(self, ellipse, replay):
        try:
            canvas = np.zeros((self.sensor_size[1], self.sensor_size[0], 3), dtype=np.uint8)
            cv2.ellipse(canvas, ellipse, (255, 255, 255), -1)
            transformed_canvas = A.ReplayCompose.replay(replay, image=canvas)["image"]
            gray = cv2.cvtColor(transformed_canvas, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None


            main_contour = max(contours, key=cv2.contourArea)


            if len(main_contour) < 5:
                return None

            new_ellipse_params = cv2.fitEllipse(main_contour)


            (x, y), (a, b), ang = new_ellipse_params
            if not all(np.isfinite([x, y, a, b, ang])):
                return None

            if a < b:
                a, b = b, a
                ang += 90
            while ang > 180:
                ang -= 360
            while ang <= -180:
                ang += 360
            x, y, a, b, ang = [round(val, 2) for val in [x, y, a, b, ang]]
            return (x, y), (a, b), ang
        except (cv2.error, ValueError):
            return None

    def cal_trig(self, ang):

        ang_rad = np.deg2rad(ang)
        return np.array([np.sin(2 * ang_rad), np.cos(2 * ang_rad)])

    def _generate_single_eye_labels(self, ellipse_aug):

        is_valid = (ellipse_aug is not None and
                    ellipse_aug[0] != (0, 0) and
                    cal_ellipse_area(ellipse_aug[1][0], ellipse_aug[1][1]) > self.pupil_area)

        close = 0 if is_valid else 1
        if not is_valid:
            ellipse_aug = ((0, 0), (0, 0), 0)

        output_h = self.default_resolution[0] // DOWN_RATIO
        output_w = self.default_resolution[1] // DOWN_RATIO

        x_orig, y_orig, a_orig, b_orig, an_orig = *ellipse_aug[0], *ellipse_aug[1], ellipse_aug[2]
        x_down, y_down, a_down, b_down = [v / DOWN_RATIO for v in [x_orig, y_orig, a_orig, b_orig]]

        x_c, y_c = np.clip(x_down, 0, output_w - 1), np.clip(y_down, 0, output_h - 1)
        a_c, b_c = np.clip(a_down, 0, output_w - 1), np.clip(b_down, 0, output_h - 1)

        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        ab = np.zeros((self.max_objs, 2), dtype=np.float32)
        ang = np.zeros((self.max_objs, 1), dtype=np.float32)
        trig = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)  # 保留reg_mask用于监督ab, trig等
        mask = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)

        if is_valid:
            center_int = np.array([x_c, y_c], dtype=np.int32)
            #radius = max(0, int(gaussian_radius((math.ceil(b_c), math.ceil(a_c)))))
            radius = max(0, int(gaussian_radius((math.ceil(b_c), math.ceil(a_c))) * 0.5))
            draw_umich_gaussian(hm[0], center_int, radius)

            ab[0] = [a_c, b_c]
            #ang[0] = an_orig + 90
            #trig[0] = self.cal_trig(an_orig + 90)
            ang_rad = np.deg2rad(an_orig)
            trig[0] = [np.cos(ang_rad), np.sin(ang_rad)]
            ind[0] = center_int[1] * output_w + center_int[0]
            reg_mask[0] = 1
            cv2.ellipse(mask[0], ((int(x_c), int(y_c)), (int(a_c), int(b_c)), int(an_orig)), 1, -1)

        # [移除] 不再返回 'reg' 标签
        return {
            "hm": torch.from_numpy(hm),
            "reg_mask": torch.from_numpy(reg_mask),
            "ind": torch.from_numpy(ind),
            "ab": torch.from_numpy(ab),
            "ang": torch.from_numpy(ang),
            "trig": torch.from_numpy(trig),
            "mask": torch.from_numpy(mask),
            "close": torch.tensor(close, dtype=torch.int64),
            "center": torch.tensor([x_down, y_down], dtype=torch.float32),
            "ellipse": torch.tensor([x_down, y_down, a_down, b_down, an_orig], dtype=torch.float32)
        }

    def __len__(self):
        return self.total_frames

    def __getitem__(self, index):

        ellipse_left_raw = convert_to_ellipse(load_ellipse(index, self.ellipse_path_left))
        event_segment_left = load_event_segment(index, self.data_path_left, self.caching_events_batch_size)

        ellipse_right_raw = convert_to_ellipse(load_ellipse(index, self.ellipse_path_right))
        event_segment_right = load_event_segment(index, self.data_path_right, self.caching_events_batch_size)


        if self.split == 'train':
            seed = random.randint(0, 2 ** 31 - 1)

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            event_transform_L = self.get_event_transforms()
            event_segment_left = event_transform_L(event_segment_left)

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            event_transform_R = self.get_event_transforms()
            event_segment_right = event_transform_R(event_segment_right)


        to_frame = lambda seg: np.moveaxis(
            to_frame_stack_numpy(seg, self.sensor_size, 1, self.events_interpolation, seg["t"][0], seg["t"][-1],
                                 10).squeeze(0),
            0, -1)
        event_frame_left = to_frame(event_segment_left)
        event_frame_right = to_frame(event_segment_right)
        cut_max_count(event_frame_left, 255)
        cut_max_count(event_frame_right, 255)


        geo_transform = self.get_transforms()
        transformed = geo_transform(image=event_frame_left, image0=event_frame_right)
        event_frame_left_aug = transformed['image']
        event_frame_right_aug = transformed['image0']
        replay_data = transformed['replay']


        ellipse_left_aug = self.transform_ellipse(ellipse_left_raw, replay_data)
        ellipse_right_aug = self.transform_ellipse(ellipse_right_raw, replay_data)

        input_frame = np.concatenate((event_frame_left_aug, event_frame_right_aug), axis=-1)
        input_tensor = torch.from_numpy(np.moveaxis(input_frame.astype(np.float32) / 255.0, -1, 0))


        labels_left = self._generate_single_eye_labels(ellipse_left_aug)
        labels_right = self._generate_single_eye_labels(ellipse_right_aug)


        ret = {'input': input_tensor}
        for key, value in labels_left.items():
            ret[key + '_L'] = value
        for key, value in labels_right.items():
            ret[key + '_R'] = value

        return ret
