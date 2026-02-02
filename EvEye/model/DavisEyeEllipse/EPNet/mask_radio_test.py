import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math
from pathlib import Path
from tqdm import tqdm
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from EvEye.model.DavisEyeEllipse.EPNet.EPNet import EPNet
from EvEye.dataset.DavisEyeEllipse.DavisEyeEllipseDataset import DavisEyeEllipseDataset
from EvEye.model.DavisEyeEllipse.EPNet.Metric import p_acc, cal_mean_distance
from EvEye.dataset.DavisEyeEllipse.utils import convert_to_ellipse

CKPT_PATH = r"D:\download\last.ckpt"
DATA_ROOT = "D:/83dataset"
OUTPUT_DIR = "D:/save_pics"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


TARGET_VIS_INDICES = []

HEAD_DICT_INTEGRAL = {
    "hm_L": 1, "ab_L": 2, "trig_L": 2, "mask_L": 1,
    "hm_R": 1, "ab_R": 2, "trig_R": 2, "mask_R": 1,
}


INPUT_RESOLUTION = (256, 256)
DOWN_RATIO = 4
SENSOR_RESOLUTION = (260, 346)

BG_COLOR = (240, 240, 240)
GT_COLOR = (0, 255, 0)
PRED_COLOR = (0, 165, 255)
SEPARATOR_COLOR = (0, 0, 0)
PANEL_SIZE = (512, 256)
POSITION_ERROR_MAGNIFICATION = 15.0
CANONICAL_MAJOR_AXIS = 150.0



def simulate_spatial_occlusion(input_tensor, mask_ratio=0.3):

    if mask_ratio <= 0.0:
        return input_tensor

    masked_tensor = input_tensor.clone()

    B, C, H, W = masked_tensor.shape


    cut_h = int(H * math.sqrt(mask_ratio))
    cut_w = int(W * math.sqrt(mask_ratio))

    cut_h = max(1, min(H, cut_h))
    cut_w = max(1, min(W, cut_w))

    y = np.random.randint(0, H - cut_h + 1)
    x = np.random.randint(0, W - cut_w + 1)

    masked_tensor[..., y:y + cut_h, x:x + cut_w] = 0

    return masked_tensor


def soft_argmax_decode(heatmap, temperature=0.1):
    B, C, H, W = heatmap.shape
    prob_dist = F.softmax(heatmap.view(B, C, -1) / temperature, dim=2).view(B, C, H, W)
    device = heatmap.device
    x_coords = torch.arange(W, device=device).float()
    y_coords = torch.arange(H, device=device).float()
    expected_x = torch.sum(prob_dist * x_coords.view(1, 1, 1, W), dim=[2, 3])
    expected_y = torch.sum(prob_dist * y_coords.view(1, 1, H, 1), dim=[2, 3])
    return torch.stack([expected_x, expected_y], dim=2)


def nms(heatmap, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    return feat.gather(1, ind)


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    return gather_feat(feat, ind)


def topk(heatmap, K=1):
    b, c, h, w = heatmap.size()
    heatmap = heatmap.view(b, c, -1)
    topk_scores, topk_inds = torch.topk(heatmap, K)
    return topk_scores, topk_inds


def restore_angle(trig):
    cosA = trig[..., 0]
    sinA = trig[..., 1]
    return torch.rad2deg(torch.atan2(sinA, cosA))


def decode_prediction(output, suffix):
    hm_pred = output[f'hm_{suffix}']
    ab_pred = output[f'ab_{suffix}']
    trig_pred = output[f'trig_{suffix}']

    coords = soft_argmax_decode(hm_pred).squeeze(0)
    xs, ys = coords[0, 0].item(), coords[0, 1].item()

    hm_sig = torch.sigmoid(hm_pred)
    hm_nms = nms(hm_sig)
    scores, inds = topk(hm_nms, K=1)

    if scores.item() < 0.1: return None

    inds = inds.view(inds.size(0), -1)
    ab = _transpose_and_gather_feat(ab_pred, inds).squeeze()
    trig = _transpose_and_gather_feat(trig_pred, inds).squeeze(0)
    ang = restore_angle(trig).item()
    a, b = abs(ab[0].item()), abs(ab[1].item())

    return ((xs, ys), (a, b), ang)



def run_evaluation_task(model, dataset, mask_ratio):


    metrics = {
        'p0.5_acc_L': [], 'p1_acc_L': [], 'p5_acc_L': [], 'p10_acc_L': [], 'dist_L': [],
        'p0.5_acc_R': [], 'p1_acc_R': [], 'p5_acc_R': [], 'p10_acc_R': [], 'dist_R': [],
    }


    for i in tqdm(range(len(dataset)), desc=f"   Eval Mask={mask_ratio:.1f}", leave=False):
        batch = dataset[i]
        input_tensor = batch['input'].unsqueeze(0).to(DEVICE)

        if mask_ratio > 0:
            input_tensor = simulate_spatial_occlusion(input_tensor, mask_ratio=mask_ratio)

        with torch.no_grad():
            pred = model(input_tensor)

        center_l_pred = soft_argmax_decode(pred['hm_L'])
        center_r_pred = soft_argmax_decode(pred['hm_R'])

        center_l_gt = batch["center_L"].unsqueeze(0).to(DEVICE)
        close_l = batch["close_L"].unsqueeze(0).to(DEVICE)
        center_r_gt = batch["center_R"].unsqueeze(0).to(DEVICE)
        close_r = batch["close_R"].unsqueeze(0).to(DEVICE)

        dets_l = {'xs': center_l_pred.squeeze(1)[:, 0:1], 'ys': center_l_pred.squeeze(1)[:, 1:2]}
        dets_r = {'xs': center_r_pred.squeeze(1)[:, 0:1], 'ys': center_r_pred.squeeze(1)[:, 1:2]}

        batch_res = {
            "p0.5_acc_L": p_acc(dets_l, center_l_gt, close_l, 0.5),
            "p1_acc_L": p_acc(dets_l, center_l_gt, close_l, 1),
            "p5_acc_L": p_acc(dets_l, center_l_gt, close_l, 5),
            "p10_acc_L": p_acc(dets_l, center_l_gt, close_l, 10),
            "dist_L": cal_mean_distance(dets_l, center_l_gt, close_l),
            "p0.5_acc_R": p_acc(dets_r, center_r_gt, close_r, 0.5),
            "p1_acc_R": p_acc(dets_r, center_r_gt, close_r, 1),
            "p5_acc_R": p_acc(dets_r, center_r_gt, close_r, 5),
            "p10_acc_R": p_acc(dets_r, center_r_gt, close_r, 10),
            "dist_R": cal_mean_distance(dets_r, center_r_gt, close_r),
        }

        for key, val in batch_res.items():
            if not torch.isnan(val):
                metrics[key].append(val.cpu().item())

    avg = {key: np.mean(val) if len(val) > 0 else 0.0 for key, val in metrics.items()}

    # 打印结果表
    print(f"   --- (Mask Ratio: {mask_ratio:.1f}) ---")
    print(f"   | Eye   | p10 Acc    | p5 Acc     | p1 Acc     | p0.5 Acc   | Mean Dist  |")
    print(f"   |-------|------------|------------|------------|------------|------------|")
    print(
        f"   | Left  | {avg['p10_acc_L']:.6f}   | {avg['p5_acc_L']:.6f}   | {avg['p1_acc_L']:.6f}   | {avg['p0.5_acc_L']:.6f}   | {avg['dist_L']:.6f}     |")
    print(
        f"   | Right | {avg['p10_acc_R']:.6f}   | {avg['p5_acc_R']:.6f}   | {avg['p1_acc_R']:.6f}   | {avg['p0.5_acc_R']:.6f}   | {avg['dist_R']:.6f}     |")
    print(f"   --------------------------------------------------------------------------")



def load_memmap(data_file, info_file):
    if not os.path.exists(info_file): return None
    with open(info_file, "r") as f:
        lines = f.readlines()
        shape = tuple(int(num) for num in lines[0].strip().split(": ")[1].strip("()").split(",") if num.strip())
        dtype = eval(lines[1].strip().split(": ")[1])
    return np.memmap(data_file, dtype=dtype, mode="r", shape=shape)


def load_ellipse_from_cache(index, ellipse_base_path):
    ellipses_path = ellipse_base_path / "ellipses_batch_0.memmap"
    ellipses_info_path = ellipse_base_path / "ellipses_batch_info_0.txt"
    if not ellipses_path.exists(): return None
    try:
        ellipses = load_memmap(ellipses_path, ellipses_info_path)
        if ellipses is not None and index < ellipses.shape[0]:
            return ellipses[index]
    except:
        pass
    return None


def draw_panel(event_frame_np, gt_ellipse, pred_ellipse):
    panel_h, panel_w = PANEL_SIZE
    top_h, top_w = INPUT_RESOLUTION
    panel = np.full((panel_h, panel_w, 3), BG_COLOR, dtype=np.uint8)

    # 1. Input View
    top_panel = np.full((top_h, top_w, 3), BG_COLOR, dtype=np.uint8)
    top_panel[(event_frame_np[:, :, 0] > 0)] = (0, 0, 255)
    top_panel[(event_frame_np[:, :, 1] > 0)] = (255, 0, 0)

    if gt_ellipse:
        (x, y), (a, b), ang = gt_ellipse
        center = (int(x * DOWN_RATIO), int(y * DOWN_RATIO))
        axes = (max(1, int(a * DOWN_RATIO)), max(1, int(b * DOWN_RATIO)))
        cv2.ellipse(top_panel, (center, axes, ang), GT_COLOR, 1, cv2.LINE_AA)
        cv2.circle(top_panel, center, 2, GT_COLOR, -1)

    if pred_ellipse:
        (x, y), (a, b), ang = pred_ellipse
        center = (int(x * DOWN_RATIO), int(y * DOWN_RATIO))
        axes = (max(1, int(a * DOWN_RATIO)), max(1, int(b * DOWN_RATIO)))
        cv2.ellipse(top_panel, (center, axes, ang), PRED_COLOR, 1, cv2.LINE_AA)
        cv2.circle(top_panel, center, 2, PRED_COLOR, -1)

    panel[0:top_h, 0:top_w] = top_panel

    # 2. Zoom View
    bottom_panel = np.full((panel_h - top_h, panel_w, 3), BG_COLOR, dtype=np.uint8)
    cx, cy = panel_w // 2, (panel_h - top_h) // 2

    if gt_ellipse:
        (xg, yg), (ag, bg), angg = gt_ellipse
        ar_g = bg / ag if ag > 1e-6 else 1.0
        axes_g = (int(CANONICAL_MAJOR_AXIS), int(CANONICAL_MAJOR_AXIS * ar_g))
        cv2.ellipse(bottom_panel, ((cx, cy), axes_g, angg), GT_COLOR, 2, cv2.LINE_AA)
        cv2.circle(bottom_panel, (cx, cy), 4, GT_COLOR, -1)

        if pred_ellipse:
            (xp, yp), (ap, bp), angp = pred_ellipse
            off_x = (xp - xg) * POSITION_ERROR_MAGNIFICATION
            off_y = (yp - yg) * POSITION_ERROR_MAGNIFICATION
            scale = ap / ag if ag > 1e-6 else 1.0
            ar_p = bp / ap if ap > 1e-6 else 1.0
            axes_p = (max(1, int(CANONICAL_MAJOR_AXIS * scale)), max(1, int(CANONICAL_MAJOR_AXIS * scale * ar_p)))
            draw_center = (int(cx + off_x), int(cy + off_y))
            cv2.ellipse(bottom_panel, (draw_center, axes_p, angp), PRED_COLOR, 2, cv2.LINE_AA)
            cv2.circle(bottom_panel, draw_center, 4, PRED_COLOR, -1)

    elif pred_ellipse:
        (xp, yp), (ap, bp), angp = pred_ellipse
        axes_p = (int(CANONICAL_MAJOR_AXIS), int(CANONICAL_MAJOR_AXIS * (bp / ap)))
        cv2.ellipse(bottom_panel, ((cx, cy), axes_p, angp), PRED_COLOR, 2, cv2.LINE_AA)

    panel[top_h:, :] = bottom_panel
    return panel


def run_visualization_task(model, dataset, mask_ratio, output_root_dir, data_root_path):
    experiment_name = f"Robust_Mask_{mask_ratio:.1f}"
    save_dir = os.path.join(output_root_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    ellipse_path_l = Path(data_root_path) / "test" / "cached_ellipse_left"
    ellipse_path_r = Path(data_root_path) / "test" / "cached_ellipse_right"
    saved_count = 0

    for i in tqdm(TARGET_VIS_INDICES, desc=f"   Vis Mask={mask_ratio:.1f}", leave=False):
        if i >= len(dataset): continue

        batch = dataset[i]
        input_tensor = batch['input'].unsqueeze(0).to(DEVICE)

        if mask_ratio > 0:
            input_tensor = simulate_spatial_occlusion(input_tensor, mask_ratio=mask_ratio)

        with torch.no_grad():
            output = model(input_tensor)

        pred_l = decode_prediction(output, 'L')
        pred_r = decode_prediction(output, 'R')

        def get_viz_gt(path_prefix, idx):
            raw = load_ellipse_from_cache(idx, path_prefix)
            if raw is None: return None
            raw_e = convert_to_ellipse(raw)
            if not raw_e: return None
            (rx, ry), (ra, rb), rang = raw_e
            sx, sy = INPUT_RESOLUTION[1] / SENSOR_RESOLUTION[1], INPUT_RESOLUTION[0] / SENSOR_RESOLUTION[0]
            return ((rx * sx / DOWN_RATIO, ry * sy / DOWN_RATIO),
                    (ra * sx / DOWN_RATIO, rb * sy / DOWN_RATIO), rang)

        gt_l_viz = get_viz_gt(ellipse_path_l, i)
        gt_r_viz = get_viz_gt(ellipse_path_r, i)

        frame_l = input_tensor[0, 0:2].cpu().permute(1, 2, 0).numpy()
        panel_l = draw_panel(frame_l, gt_l_viz, pred_l)

        frame_r = input_tensor[0, 2:4].cpu().permute(1, 2, 0).numpy()
        panel_r = draw_panel(frame_r, gt_r_viz, pred_r)

        final_img = np.hstack([panel_l, panel_r])
        cv2.line(final_img, (PANEL_SIZE[1], 0), (PANEL_SIZE[1], PANEL_SIZE[0]), SEPARATOR_COLOR, 2)

        cv2.imwrite(os.path.join(save_dir, f"{i:05d}.png"), final_img)
        saved_count += 1

    print(f"   完成。已保存 {saved_count} 张图片。")



def execute_experiment(model, dataset, mask_ratio, args):

    if not args.skip_eval:
        run_evaluation_task(model, dataset, mask_ratio)

    if not args.skip_viz:
        run_visualization_task(model, dataset, mask_ratio, args.out, args.data)


def main():
    parser = argparse.ArgumentParser(description="EPNet Occlusion Robustness")
    parser.add_argument('--ckpt', type=str, default=CKPT_PATH)
    parser.add_argument('--data', type=str, default=DATA_ROOT)
    parser.add_argument('--out', type=str, default=OUTPUT_DIR)
    parser.add_argument('--cache-bs', type=int, default=5000)


    parser.add_argument('--batch-test', action='store_true', help='0.1 - 0.9')
    parser.add_argument('--single-mask', type=float, default=0.0)


    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--skip-viz', action='store_true')

    args = parser.parse_args()

    model = EPNet(input_channels=4, head_dict=HEAD_DICT_INTEGRAL, fpn_out_channels=64)
    if not os.path.exists(args.ckpt): print(f"Error: {args.ckpt}"); return
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE)["state_dict"])
    model.eval().to(DEVICE)

    val_dataset = DavisEyeEllipseDataset(root_path=args.data, split="test", caching_events_batch_size=args.cache_bs,
                                         default_resolution=INPUT_RESOLUTION)
    if len(val_dataset) == 0: return

    if args.batch_test:
        mask_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for ratio in mask_ratios:
            execute_experiment(model, val_dataset, ratio, args)
    else:
        execute_experiment(model, val_dataset, args.single_mask, args)



if __name__ == '__main__':
    main()