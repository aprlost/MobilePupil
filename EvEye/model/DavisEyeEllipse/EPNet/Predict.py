

import torch
import numpy as np
import albumentations as A
import time
import random
import cv2
import os
from tqdm import tqdm
from thop import profile
from pathlib import Path
from torch.nn import functional as F

# from EvEye.model.DavisEyeEllipse.EPNet.EPNet import EPNet
from EvEye.utils.tonic.functional.ToFrameStack import to_frame_stack_numpy
from EvEye.utils.cache.MemmapCacheStructedEvents import *
from EvEye.utils.visualization.visualization import *
from EvEye.utils.tonic.functional.CutMaxCount import cut_max_count
from EvEye.dataset.DavisEyeEllipse.utils import *

torch.set_printoptions(sci_mode=False)


def soft_argmax_decode(heatmap, temperature=1.0):

    B, C, H, W = heatmap.shape
    heatmap_flat = heatmap.view(B, C, -1)
    prob_dist = F.softmax(heatmap_flat / temperature, dim=2).view(B, C, H, W)

    device = heatmap.device
    dtype = heatmap.dtype
    x_coords = torch.arange(W, device=device, dtype=dtype).float()
    y_coords = torch.arange(H, device=device, dtype=dtype).float()

    expected_x = torch.sum(prob_dist * x_coords.view(1, 1, 1, W), dim=[2, 3])
    expected_y = torch.sum(prob_dist * y_coords.view(1, 1, H, 1), dim=[2, 3])

    coords = torch.stack([expected_x, expected_y], dim=2)
    return coords


def nms(heatmap, kernel=3):
    pad = (kernel - 1) // 2
    hmax = torch.nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def transpose_feat(feat):
    b, c, h, w = feat.size()
    feat = feat.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
    return feat


def gather_feat(feat, ind):
    b, _, c = feat.size()
    ind = ind.unsqueeze(2).expand(b, ind.size(1), c)
    return feat.gather(1, ind)


def topk(heatmap, K=1):
    b, c, h, w = heatmap.size()
    heatmap = heatmap.view(b, c, -1)
    topk_scores, topk_inds = torch.topk(heatmap, K)

    topk_clses = torch.zeros_like(topk_scores, dtype=torch.long)
    topk_ys = (topk_inds / w).int().float()
    topk_xs = (topk_inds % w).int().float()

    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def restore_angle(cos_sin_vec: torch.tensor):

    cos_sin_vec = cos_sin_vec.squeeze(1)
    cosA = cos_sin_vec[:, 0]
    sinA = cos_sin_vec[:, 1]
    return torch.rad2deg(torch.atan2(sinA, cosA))


def get_ellipse_from_tensors(xs, ys, ab, ang, score, score_threshold=0.1):

    if score.item() < score_threshold:
        return ((0, 0), (0, 0), 0)

    x = xs.item()
    y = ys.item()
    a = ab[0, 0].item()
    b = ab[0, 1].item()

    angle = ang.item()

    raw_ellipse = ((x, y), (a, b), angle)
    return raw_ellipse


def transform_ellipse(raw_ellipse, orig_size=(64, 64), target_size=(260, 346)):

    scale_y = target_size[0] / orig_size[0]
    scale_x = target_size[1] / orig_size[1]

    (x, y), (a, b), ang = raw_ellipse

    new_x = x * scale_x
    new_y = y * scale_y
    new_a = a * scale_x
    new_b = b * scale_y

    return ((new_x, new_y), (new_a, new_b), ang)


def _transpose_and_gather_feat(feat, ind):

    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat


def predict(model, model_path, data_path_left, ellipse_path_left, data_path_right, ellipse_path_right, output_path,
            num=None, device="cuda:0"):

    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()
    model.to(device)

    total_frames = get_nums(ellipse_path_left) if num is None else min(get_nums(ellipse_path_left), num)

    for index in tqdm(range(total_frames), desc="Predicting"):
        event_segment_left = load_event_segment(index, data_path_left, 5000)
        event_segment_right = load_event_segment(index, data_path_right, 5000)

        to_frame = lambda seg: np.moveaxis(
            to_frame_stack_numpy(seg, (346, 260, 2), 1, "causal_linear", seg["t"][0], seg["t"][-1], 10).squeeze(0), 0,
            -1)
        frame_left = to_frame(event_segment_left)
        frame_right = to_frame(event_segment_right)
        cut_max_count(frame_left, 255)
        cut_max_count(frame_right, 255)

        transform = A.Resize(256, 256)
        frame_left = transform(image=frame_left)['image']
        frame_right = transform(image=frame_right)['image']

        input_frame = np.concatenate((frame_left, frame_right), axis=-1)
        input_tensor = torch.from_numpy(np.moveaxis(input_frame.astype(np.float32) / 255.0, -1, 0)).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)

        ellipses_to_draw = {}
        for suffix in ['L', 'R']:
            hm_pred = output[f'hm_{suffix}']
            ab_pred = output[f'ab_{suffix}']
            trig_pred = output[f'trig_{suffix}']

            coords = soft_argmax_decode(hm_pred, temperature=0.1)
            xs = coords[:, 0, 0:1]  # Shape: (1, 1)
            ys = coords[:, 0, 1:2]  # Shape: (1, 1)

            hm_sig = torch.sigmoid(hm_pred)
            hm_nms = nms(hm_sig)
            scores, inds, _, _, _ = topk(hm_nms, K=1)

            ab = _transpose_and_gather_feat(ab_pred, inds)  # Shape: (1, 1, 2)
            trig = _transpose_and_gather_feat(trig_pred, inds)  # Shape: (1, 1, 2)
            ang = restore_angle(trig)  # Shape: (1)

            raw_ellipse_low_res = get_ellipse_from_tensors(xs, ys, ab, ang, scores)

            down_ratio = 4
            orig_size = (256 // down_ratio, 256 // down_ratio)
            final_ellipse = transform_ellipse(raw_ellipse_low_res, orig_size=orig_size, target_size=(260, 346))
            ellipses_to_draw[suffix] = final_ellipse

        vis_frame = visualizeHWC(frame_left, normalized=True, add_weight=10)

        if ellipses_to_draw['L'][1] != (0, 0):
            cv2.ellipse(vis_frame, ellipses_to_draw['L'], (255, 0, 0), 2)

        if ellipses_to_draw['R'][1] != (0, 0):
            cv2.ellipse(vis_frame, ellipses_to_draw['R'], (0, 255, 0), 2)

        save_path = os.path.join(output_path, f"{index:05}.png")
        save_image(vis_frame, save_path)

def get_nums(ellipse_path):
    num_frames_list = []
    ellipses_list = load_cached_structed_ellipses(Path(ellipse_path))
    for ellipses in ellipses_list:
        num_frames_list.append(len(ellipses))
    total_frames = sum(num_frames_list)
    return total_frames


def test_inference_time(model, model_path, device="cuda:0"):

    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()
    model.to(device)

    input_tensor = torch.rand((1, 4, 256, 256), dtype=torch.float32).to(device)
    with torch.no_grad():
        times = []
        # Warm-up
        for _ in range(50):
            model(input_tensor)

        for _ in tqdm(range(200), desc="Testing inference time"):
            torch.cuda.synchronize()
            start_time = time.time()
            output = model(input_tensor)
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time * 1000:.2f} ms")
    return avg_time


def main():

    from EvEye.model.DavisEyeEllipse.EPNet.EPNet import EPNet

    HEAD_DICT_INTEGRAL = {
        "hm_L": 1, "ab_L": 2, "trig_L": 2, "mask_L": 1,
        "hm_R": 1, "ab_R": 2, "trig_R": 2, "mask_R": 1,
    }
    model = EPNet(
        input_channels=4,
        head_dict=HEAD_DICT_INTEGRAL,
        mode="fpn_2d",
    )

    base_path = Path("dataset/val")
    model_path = "checkpoint.ckpt"
    output_path = Path("./predictions_output")
    os.makedirs(output_path, exist_ok=True)

    data_path_left = base_path / "cached_data_left"
    ellipse_path_left = base_path / "cached_ellipse_left"
    data_path_right = base_path / "cached_data_right"
    ellipse_path_right = base_path / "cached_ellipse_right"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    predict(
        model=model,
        model_path=model_path,
        data_path_left=data_path_left,
        ellipse_path_left=ellipse_path_left,
        data_path_right=data_path_right,
        ellipse_path_right=ellipse_path_right,
        output_path=output_path,
        num=200,
        device=device
    )
if __name__ == "__main__":
    main()