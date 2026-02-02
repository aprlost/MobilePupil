import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


def rand_range(amin, amax):
    return amin + (amax - amin) * torch.rand(1).item()


class TorchEventFrameRandomAffine:
    def __init__(
            self,
            size=(260, 346),
            translate=(0.2, 0.2),
            degrees=15,
            scale=(0.8, 1.2),
            spatial_jitter=None,
    ):
        self.height, self.width = size
        self.translate = translate
        self.degrees = degrees
        self.scale = scale
        self.spatial_jitter = spatial_jitter
        self.affine_matrix = self.get_affine_matrix()

    def normalize(self, coords, backward=False):
        if not backward:
            coords[0] = coords[0] / (self.width - 1) - 0.5
            coords[1] = coords[1] / (self.height - 1) - 0.5
        else:
            coords[0] = (coords[0] + 0.5) * (self.width - 1)
            coords[1] = (coords[1] + 0.5) * (self.height - 1)
        return coords

    def get_translation_matrix(self):
        translate = [rand_range(-t, t) for t in self.translate]
        return torch.tensor(
            [[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]],
            dtype=torch.float32
        )

    def get_rotation_matrix(self):
        degrees = rand_range(-self.degrees, self.degrees) / 180 * np.pi
        cos, sin = torch.cos(torch.tensor(degrees)), torch.sin(torch.tensor(degrees))
        return torch.tensor(
            [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]],
            dtype=torch.float32
        )

    def get_scale_matrix(self):
        scale = [rand_range(*self.scale) for _ in range(2)]
        return torch.tensor(
            [[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]],
            dtype=torch.float32
        )

    def get_affine_matrix(self):
        T = self.get_translation_matrix()
        R = self.get_rotation_matrix()
        S = self.get_scale_matrix()
        return T @ R @ S

    def transform_event_frame(self, event_frame: torch.Tensor):
        """
        支持输入：
        - 3维：[C, H, W]（单时间步）
        - 4维：[C, T, H, W]（多时间步）
        输出保持同维度，保留所有通道信息。
        """
        original_ndim = event_frame.ndim
        assert original_ndim in (3, 4), f"输入维度应为3或4，实际为{original_ndim}"

        # 统一转换为 [T, C, H, W] 格式
        if original_ndim == 4:
            event_frame = event_frame.permute(1, 0, 2, 3)  # (C, T, H, W) → (T, C, H, W)
        else:  # 3维，假设是 [C, H, W]（T=1）
            event_frame = event_frame.unsqueeze(0)  # → [1, C, H, W]

        t, c, h, w = event_frame.shape

        # 构建仿射矩阵（适配batch）
        affine_matrix = self.affine_matrix.to(event_frame.device)
        affine_matrix = affine_matrix.unsqueeze(0).repeat(t * c, 1, 1)
        affine_matrix_2x3 = affine_matrix[:, :2, :]  # 提取2x3部分（affine_grid要求）

        # 生成网格并变换
        grid = F.affine_grid(affine_matrix_2x3, [t * c, 1, h, w], align_corners=False)
        event_frame_flat = event_frame.view(t * c, 1, h, w)
        event_frame_flat = F.grid_sample(event_frame_flat, grid, align_corners=False)
        event_frame = event_frame_flat.view(t, c, h, w)

        # 恢复原始维度
        if original_ndim == 4:
            event_frame = event_frame.permute(1, 0, 2, 3)  # → (C, T, H, W)
        else:  # 3维，还原为 [C, H, W]
            event_frame = event_frame.squeeze(0)

        return event_frame

    def transform_label(self, label: torch.Tensor):
        assert label.shape[0] == 2 and label.ndim == 2
        label_ones = torch.ones((1, label.shape[1]))
        label_stacked = torch.cat((label, label_ones), dim=0)
        label_affined = torch.mm(self.affine_matrix, label_stacked)
        return label_affined[:2].round()

    def temporal_flip(
            self, event_frame: torch.Tensor, label: torch.Tensor, closeness: torch.Tensor
    ):
        assert event_frame.ndim == 4 and event_frame.shape[0] == 2
        assert label.ndim == 2 and label.shape[0] == 2
        assert closeness.ndim == 1 and closeness.shape[0] == label.shape[1]
        event_frame = torch.flip(event_frame, dims=[0, 1])
        label = torch.flip(label, dims=[1])
        closeness = torch.flip(closeness, dims=[0])
        return event_frame, label, closeness


def read_image_to_tensor(image_path):
    """
    读取PNG图像，保留RGB三通道：
    - 输入：PNG文件（RGB三通道）
    - 输出：[3, H, W] 张量（通道0=R，通道1=G，通道2=B）
    """
    image = Image.open(image_path).convert('RGB')  # 强制转为RGB，确保3通道
    image_np = np.array(image)
    # 转换为 [C, H, W]
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
    return image_tensor


def save_tensor_as_png(tensor, save_path):
    """
    将3通道张量保存为RGB彩色图：
    - 通道0 → R
    - 通道1 → G
    - 通道2 → B
    """
    assert tensor.ndim == 3 and tensor.shape[0] == 3, "张量需为[3, H, W]格式"
    # 分离通道并转换为numpy数组
    r_channel = tensor[0].cpu().numpy()
    g_channel = tensor[1].cpu().numpy()
    b_channel = tensor[2].cpu().numpy()

    # 裁剪到0-255范围，转换为uint8
    r = np.clip(r_channel, 0, 255).astype(np.uint8)
    g = np.clip(g_channel, 0, 255).astype(np.uint8)
    b = np.clip(b_channel, 0, 255).astype(np.uint8)

    # 合并为RGB图像
    rgb_image = np.stack([r, g, b], axis=2)
    Image.fromarray(rgb_image).save(save_path)


def process_images(root_path, save_dir):
    """
    批量处理图像：
    1. 读取PNG并转为3通道张量
    2. 应用仿射变换
    3. 保存为RGB彩色PNG
    """
    augmenter = TorchEventFrameRandomAffine()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for filename in os.listdir(root_path):
        if filename.endswith('.png'):
            image_path = os.path.join(root_path, filename)
            # 读取图像（[3, H, W]）
            event_frame = read_image_to_tensor(image_path)
            # 增广
            augmented_frame = augmenter.transform_event_frame(event_frame)
            # 保存
            save_filename = f"aug_{filename}"
            save_path = os.path.join(save_dir, save_filename)
            save_tensor_as_png(augmented_frame, save_path)
            print(f"✅ 保存增广图像：{save_path}")





if __name__ == "__main__":
    root_path = "D:/make_dataset/user1/right/session_2_0_1/events/outframes/events"  # 替换为实际的根目录路径
    save_dir = "D:/make_dataset/user1/right/session_2_0_1/events/zengguang"    # 替换为实际的保存目录路径
    process_images(root_path, save_dir)
