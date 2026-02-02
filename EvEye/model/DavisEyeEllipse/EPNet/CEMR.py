import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn_act(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True)
    )

class MSDA(nn.Module):


    def __init__(self, in_channels, reduction=4):
        super(MSDA, self).__init__()
        mid_channels = in_channels // reduction

        self.local_branch = nn.Conv2d(in_channels, mid_channels, 3, padding=1, dilation=1, groups=mid_channels)
        self.global_branch = nn.Conv2d(in_channels, mid_channels, 3, padding=2, dilation=2,
                                       groups=mid_channels)  # Dilation=2
        self.wide_branch = nn.Conv2d(in_channels, mid_channels, 3, padding=4, dilation=4,
                                     groups=mid_channels)  # Dilation=4

        self.attention_fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 3, in_channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        local_feat = self.local_branch(x)
        global_feat = self.global_branch(x)
        wide_feat = self.wide_branch(x)

        stacked = torch.cat([local_feat, global_feat, wide_feat], dim=1)

        attn_map = self.attention_fusion(stacked)
        return x * attn_map


class SAU(nn.Module):


    def __init__(self, in_channels, scale_factor=2, k_encoder=3, k_up=5):
        super(SAU, self).__init__()
        self.scale_factor = scale_factor
        self.k_up = k_up

        self.content_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=k_encoder, padding=k_encoder // 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )


        self.kernel_predictor = nn.Conv2d(64, (scale_factor * k_up) ** 2, kernel_size=1)

        self.gate_predictor = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, (scale_factor * k_up) ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        N, C, H, W = x.size()
        S = self.scale_factor
        K = self.k_up

        content = self.content_encoder(x)

        kernel = self.kernel_predictor(content)
        gate = self.gate_predictor(content)

        kernel = kernel * gate
        kernel = F.softmax(kernel.view(N, (S * K) ** 2, H * W), dim=1)
        kernel = kernel.view(N, 1, K, K, S, S, H, W)


        x_pad = F.pad(x, (K // 2, K // 2, K // 2, K // 2), mode='replicate')
        x_unfold = F.unfold(x_pad, kernel_size=K, padding=0, stride=1)
        x_unfold = x_unfold.view(N, C, K, K, H, W)


        output = x_unfold.unsqueeze(4).unsqueeze(4) * kernel

        output = output.sum(dim=(2, 3))
        output = output.permute(0, 1, 4, 2, 5, 3).reshape(N, C, H * S, W * S)

        return output



class CEMR(nn.Module):


    def __init__(self, in_channels_list, out_channels=64):
        super(CEMR, self).__init__()

        self.lateral_p3 = conv_bn_act(in_channels_list[0], out_channels, 1)
        self.lateral_p4 = conv_bn_act(in_channels_list[1], out_channels, 1)
        self.lateral_p5 = conv_bn_act(in_channels_list[2], out_channels, 1)

        self.sau_p5_to_p4 = SAU(out_channels, scale_factor=2)
        self.sau_p4_to_p3 = SAU(out_channels, scale_factor=2)

        self.smooth_p4 = conv_bn_act(out_channels, out_channels, 3)
        self.smooth_p3 = conv_bn_act(out_channels, out_channels, 3)

        self.refine_p3 = MSDA(out_channels)
        self.refine_p4 = MSDA(out_channels)
        self.refine_p5 = MSDA(out_channels)

    def forward(self, p3, p4, p5):

        p5_lat = self.lateral_p5(p5)
        p4_lat = self.lateral_p4(p4)
        p3_lat = self.lateral_p3(p3)


        # Level 5
        p5_out = self.refine_p5(p5_lat)

        # Level 4: P5_up + P4
        p5_up = self.sau_p5_to_p4(p5_out)

        if p5_up.shape[-2:] != p4_lat.shape[-2:]:
            p5_up = F.interpolate(p5_up, size=p4_lat.shape[-2:], mode='bilinear', align_corners=False)

        p4_fused = self.smooth_p4(p4_lat + p5_up)  # Add fusion
        p4_out = self.refine_p4(p4_fused)  # MSDA Refine

        # Level 3: P4_up + P3
        p4_up = self.sau_p4_to_p3(p4_out)
        if p4_up.shape[-2:] != p3_lat.shape[-2:]:
            p4_up = F.interpolate(p4_up, size=p3_lat.shape[-2:], mode='bilinear', align_corners=False)

        p3_fused = self.smooth_p3(p3_lat + p4_up)  # Add fusion
        p3_out = self.refine_p3(p3_fused)  # MSDA Refine

        return p3_out, p4_out, p5_out