
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class MMSEFusionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=4):

        super(MMSEFusionBlock, self).__init__()


        self.variance_predictor = nn.Sequential(
            nn.Conv2d(in_channels, max(1, in_channels // reduction), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, max(1, in_channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, kernel_size=3, padding=1, bias=False),
            nn.Softplus()
        )

        self.post_fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            h_swish()
        )

    def forward(self, x_left, x_right):

        var_left = self.variance_predictor(x_left)
        var_right = self.variance_predictor(x_right)

        eps = 1e-6
        denominator = var_left + var_right + eps

        weight_left = var_right / denominator
        weight_right = var_left / denominator

        fused_feat = weight_left * x_left + weight_right * x_right

        out = self.post_fusion(fused_feat)

        return out


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out



class DualPathStem(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DualPathStem, self).__init__()
        assert input_channels == 4, "DualPathStem requires 4 input channels"

        self.mid_channels = output_channels // 2

        self.left_eye_conv = nn.Sequential(
            nn.Conv2d(2, self.mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            h_swish()
        )
        self.right_eye_conv = nn.Sequential(
            nn.Conv2d(2, self.mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            h_swish()
        )


    def forward(self, x):
        return None



class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup

        attention_layer = CoordAtt(hidden_dim, hidden_dim)

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # attention
                attention_layer,
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                # attention
                attention_layer,
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)



class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=nn.BatchNorm2d)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1],
                                 BatchNorm=nn.BatchNorm2d)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2],
                                 BatchNorm=nn.BatchNorm2d)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3],
                                 BatchNorm=nn.BatchNorm2d)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(256 * 5, outplanes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class MobileNetV3Backbone(nn.Module):
    def __init__(self, input_channels=4, width_mult=1.0, aspp_out_channels=256):
        super(MobileNetV3Backbone, self).__init__()

        self.in_filters = [
            _make_divisible(40 * width_mult, 8),  # P3
            _make_divisible(112 * width_mult, 8),  # P4
            aspp_out_channels  # P5
        ]

        self.stage_out_indices = [5, 11, 14]

        self.cfgs = [
            # k, t, c, HS, s
            [3, 1, 16, 0, 1],
            [3, 4.5, 24, 0, 2],
            [3, 3.7, 24, 0, 1],
            [5, 4, 40, 0, 2],
            [5, 4, 40, 0, 1],
            [5, 4, 40, 0, 1],  # index 5, P3
            [3, 6, 80, 1, 2],
            [3, 2.5, 80, 1, 1],
            [3, 2.3, 80, 1, 1],
            [3, 2.3, 80, 1, 1],
            [3, 6, 112, 1, 1],  # index 10
            [3, 6, 112, 1, 1],  # index 11, P4
            [5, 6, 160, 1, 2],
            [5, 6, 160, 1, 1],
            [5, 6, 160, 1, 1],  # index 14, P5
        ]

        init_channels = _make_divisible(16 * width_mult, 8)

        self.stem = DualPathStem(input_channels, init_channels)

        self.mmse_fusion = MMSEFusionBlock(
            in_channels=init_channels // 2,
            out_channels=init_channels
        )

        self.blocks = nn.ModuleList()
        input_channel = init_channels
        for k, t, c, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            self.blocks.append(
                InvertedResidual(input_channel, exp_size, output_channel, k, s, use_hs)
            )
            input_channel = output_channel

        last_stage_inplanes = _make_divisible(self.cfgs[-1][2] * width_mult, 8)
        self.aspp = ASPP(inplanes=last_stage_inplanes, outplanes=aspp_out_channels)

    def forward(self, x):

        left_feat_raw = self.stem.left_eye_conv(x[:, 0:2, :, :])  # shape: [B, C/2, H, W]
        right_feat_raw = self.stem.right_eye_conv(x[:, 2:4, :, :])  # shape: [B, C/2, H, W]


        x = self.mmse_fusion(left_feat_raw, right_feat_raw)

        outputs = []


        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.stage_out_indices:
                outputs.append(x)


        outputs[-1] = self.aspp(outputs[-1])

        return outputs[0], outputs[1], outputs[2]


def main():
    model = MobileNetV3Backbone(input_channels=4, aspp_out_channels=256)

    # (L_on, L_off, R_on, R_off)，128x128
    input_tensor = torch.randn(2, 4, 128, 128)

    try:

        p3, p4, p5 = model(input_tensor)

        print(f"P3 output shape: {p3.shape}")  # H/4
        print(f"P4 output shape: {p4.shape}")  # H/8
        print(f"P5 output shape: {p5.shape}")  # H/16



    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()