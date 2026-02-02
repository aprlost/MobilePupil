import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time
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

CKPT_PATH = ""
DATA_ROOT = "D:/83dataset"
OUTPUT_DIR = "D:/viz"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

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



def simulate_event_drop(input_tensor, drop_rate=0.5):
    if drop_rate <= 0.0: return input_tensor
    keep_prob = 1.0 - drop_rate
    mask = torch.bernoulli(torch.full_like(input_tensor, keep_prob))
    return input_tensor * mask


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




def run_evaluation(model, dataset, args):

    mode_str = "Standard" if args.mode == 0 else f"Robust_Drop{args.drop_rate}"


    model.eval()
    model.to(DEVICE)

    metrics = {
        'p1_acc_L': [], 'p5_acc_L': [], 'p10_acc_L': [], 'dist_L': [], 'p0.5_acc_L': [],
        'p1_acc_R': [], 'p5_acc_R': [], 'p10_acc_R': [], 'dist_R': [], 'p0.5_acc_R': [],
    }

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            batch = dataset[i]
            input_tensor = batch['input'].unsqueeze(0).to(DEVICE)
            if args.mode == 1:
                input_tensor = simulate_event_drop(input_tensor, drop_rate=args.drop_rate)

            pred = model(input_tensor)
            center_l_pred = soft_argmax_decode(pred['hm_L'])
            center_r_pred = soft_argmax_decode(pred['hm_R'])

            center_l_gt = batch["center_L"].unsqueeze(0).to(DEVICE)
            close_l = batch["close_L"].unsqueeze(0).to(DEVICE)
            center_r_gt = batch["center_R"].unsqueeze(0).to(DEVICE)
            close_r = batch["close_R"].unsqueeze(0).to(DEVICE)

            dets_l = {'xs': center_l_pred.squeeze(1)[:, 0:1], 'ys': center_l_pred.squeeze(1)[:, 1:2]}
            dets_r = {'xs': center_r_pred.squeeze(1)[:, 0:1], 'ys': center_r_pred.squeeze(1)[:, 1:2]}

            batch_metrics = {
                "p1_acc_L": p_acc(dets_l, center_l_gt, close_l, 1),
                "p5_acc_L": p_acc(dets_l, center_l_gt, close_l, 5),
                "p10_acc_L": p_acc(dets_l, center_l_gt, close_l, 10),
                "p0.5_acc_L": p_acc(dets_l, center_l_gt, close_l, 0.5),
                "dist_L": cal_mean_distance(dets_l, center_l_gt, close_l),
                "p1_acc_R": p_acc(dets_r, center_r_gt, close_r, 1),
                "p5_acc_R": p_acc(dets_r, center_r_gt, close_r, 5),
                "p10_acc_R": p_acc(dets_r, center_r_gt, close_r, 10),
                "p0.5_acc_R": p_acc(dets_r, center_r_gt, close_r, 0.5),
                "dist_R": cal_mean_distance(dets_r, center_r_gt, close_r),
            }

            for key, value in batch_metrics.items():
                if not torch.isnan(value):
                    metrics[key].append(value.cpu().item())


    avg = {key: np.mean(val) if len(val) > 0 else 0 for key, val in metrics.items()}


    print(f"  --- Left ---")
    print(
        f"    - p10/p5/p1/p0.5 Acc: {avg['p10_acc_L']:.4f} / {avg['p5_acc_L']:.4f} / {avg['p1_acc_L']:.4f} / {avg['p0.5_acc_L']:.4f}")
    print(f"    - Dist: {avg['dist_L']:.4f}")
    print(f"  --- Right ---")
    print(
        f"    - p10/p5/p1/p0.5 Acc: {avg['p10_acc_R']:.4f} / {avg['p5_acc_R']:.4f} / {avg['p1_acc_R']:.4f} / {avg['p0.5_acc_R']:.4f}")
    print(f"    - Dist: {avg['dist_R']:.4f}")

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
    ellipses = load_memmap(ellipses_path, ellipses_info_path)
    return ellipses[index] if ellipses is not None else None


def draw_panel(event_frame_np, gt_ellipse, pred_ellipse):

    panel_h, panel_w = PANEL_SIZE
    top_h, top_w = INPUT_RESOLUTION

    panel = np.full((panel_h, panel_w, 3), BG_COLOR, dtype=np.uint8)
    top_panel_img = np.full((top_h, top_w, 3), BG_COLOR, dtype=np.uint8)

    top_panel_img[(event_frame_np[:, :, 0] > 0)] = (0, 0, 255)
    top_panel_img[(event_frame_np[:, :, 1] > 0)] = (255, 0, 0)

    for ellipse, color in [(gt_ellipse, GT_COLOR), (pred_ellipse, PRED_COLOR)]:
        if ellipse:
            (x, y), (a, b), ang = ellipse
            center = (int(x * DOWN_RATIO), int(y * DOWN_RATIO))
            axes = (max(1, int(a * DOWN_RATIO)), max(1, int(b * DOWN_RATIO)))
            cv2.ellipse(top_panel_img, (center, axes, ang), color, 1, cv2.LINE_AA)
            cv2.circle(top_panel_img, center, 2, color, -1)

    panel[0:top_h, 0:top_w] = top_panel_img

    bottom_panel_img = np.full((panel_h - top_h, panel_w, 3), BG_COLOR, dtype=np.uint8)
    zoom_cx, zoom_cy = panel_w // 2, (panel_h - top_h) // 2

    if gt_ellipse:
        (x_gt, y_gt), (a_gt, b_gt), ang_gt = gt_ellipse
        gt_ar = b_gt / a_gt if a_gt > 1e-6 else 1.0
        gt_axes_z = (int(CANONICAL_MAJOR_AXIS), int(CANONICAL_MAJOR_AXIS * gt_ar))

        cv2.ellipse(bottom_panel_img, ((zoom_cx, zoom_cy), gt_axes_z, ang_gt), GT_COLOR, 2, cv2.LINE_AA)
        cv2.circle(bottom_panel_img, (zoom_cx, zoom_cy), 4, GT_COLOR, -1)

        if pred_ellipse:
            (x_p, y_p), (a_p, b_p), ang_p = pred_ellipse

            off_x = (x_p - x_gt) * POSITION_ERROR_MAGNIFICATION
            off_y = (y_p - y_gt) * POSITION_ERROR_MAGNIFICATION

            scale = a_p / a_gt if a_gt > 1e-6 else 1.0
            ar_p = b_p / a_p if a_p > 1e-6 else 1.0
            pred_axes_z = (max(1, int(CANONICAL_MAJOR_AXIS * scale)), max(1, int(CANONICAL_MAJOR_AXIS * scale * ar_p)))

            draw_center = (int(zoom_cx + off_x), int(zoom_cy + off_y))
            cv2.ellipse(bottom_panel_img, (draw_center, pred_axes_z, ang_p), PRED_COLOR, 2, cv2.LINE_AA)
            cv2.circle(bottom_panel_img, draw_center, 4, PRED_COLOR, -1)

    panel[top_h:, :] = bottom_panel_img
    return panel


def run_visualization(model, dataset, args):
    mode_str = "Standard" if args.mode == 0 else f"Robust_Drop{args.drop_rate}"
    output_dir = os.path.join(args.out, mode_str)
    print(f"--- 3. 生成可视化结果 [{mode_str}] ---")
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    model.to(DEVICE)

    ellipse_path_l = Path(args.data) / "test" / "cached_ellipse_left"
    ellipse_path_r = Path(args.data) / "test" / "cached_ellipse_right"

    total_frames = min(len(dataset), args.num_viz)
    saved_count = 0

    for i in tqdm(range(total_frames), desc="  - 可视化进度"):
        batch = dataset[i]
        input_tensor = batch['input'].unsqueeze(0).to(DEVICE)

        if args.mode == 1:
            input_tensor = simulate_event_drop(input_tensor, drop_rate=args.drop_rate)

        with torch.no_grad():
            output = model(input_tensor)

        pred_l = decode_prediction(output, 'L')
        pred_r = decode_prediction(output, 'R')

        if pred_l is not None and pred_r is not None:

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
            cv2.imwrite(os.path.join(output_dir, f"{i:05d}.png"), final_img)
            saved_count += 1


def main():
    parser = argparse.ArgumentParser(description="EPNet test")
    parser.add_argument('--ckpt', type=str, default=CKPT_PATH)
    parser.add_argument('--data', type=str, default=DATA_ROOT)
    parser.add_argument('--out', type=str, default=OUTPUT_DIR)
    parser.add_argument('--num-viz', type=int, default=0)
    parser.add_argument('--cache-bs', type=int, default=5000)
    parser.add_argument('--mode', type=int, default=1, choices=[0, 1], help='0=Normal, 1=DropEvent')
    parser.add_argument('--drop-rate', type=float, default=0.3)
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--skip-viz', action='store_true')

    args = parser.parse_args()

    model = EPNet(input_channels=4, head_dict=HEAD_DICT_INTEGRAL, fpn_out_channels=64)
    if not os.path.exists(args.ckpt):
        return
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE)["state_dict"])


    val_dataset = DavisEyeEllipseDataset(root_path=args.data, split="test", caching_events_batch_size=args.cache_bs,
                                         default_resolution=INPUT_RESOLUTION)


    if not args.skip_eval:
        run_evaluation(model, val_dataset, args)
        print(1)

    if not args.skip_viz:
        run_visualization(model, val_dataset, args)

if __name__ == '__main__':
    main()