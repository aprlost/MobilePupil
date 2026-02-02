

import torch
import torch.nn as nn


class EPHead_v2(nn.Module):

    def __init__(self, in_channels, head_conv=512, head_dict=None):

        super(EPHead_v2, self).__init__()

        if head_dict is None:
            raise ValueError("head_dict 必须被提供")
        self.heads = head_dict

        self.shared_trunk = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_conv),
            nn.SiLU(inplace=True)
        )

        for head in self.heads:
            classes = self.heads[head]

            final_layer = nn.Conv2d(
                in_channels=head_conv,
                out_channels=classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )

            if 'hm' in head:
                final_layer.bias.data.fill_(-2.19)
            else:

                nn.init.constant_(final_layer.bias, 0)

            self.__setattr__(head, final_layer)

    def forward(self, x):

        common_features = self.shared_trunk(x)

        outputs = {}

        for head in self.heads:

            final_layer = self.__getattr__(head)
            raw_output = final_layer(common_features)

            if 'trig' in head:
                final_output = torch.tanh(raw_output)
            else:
                final_output = raw_output

            outputs[head] = final_output

        return outputs


def main():

    HEAD_DICT_INTEGRAL = {
        "hm_L": 1, "ab_L": 2, "trig_L": 2, "mask_L": 1,
        "hm_R": 1, "ab_R": 2, "trig_R": 2, "mask_R": 1,
    }


    batch_size = 4
    fpn_out_channels = 64
    head_conv_channels = 512
    height, width = 64, 64

    input_tensor = torch.randn(batch_size, fpn_out_channels, height, width)

    try:
        head_v2 = EPHead_v2(
            in_channels=fpn_out_channels,
            head_conv=head_conv_channels,
            head_dict=HEAD_DICT_INTEGRAL
        )

        print(head_v2)
        output = head_v2(input_tensor)

        for head_name, tensor in output.items():
            expected_channels = HEAD_DICT_INTEGRAL[head_name]
            print(f"  - {head_name:<7}: Shape: {tensor.shape}, Channels: {expected_channels}")
            assert tensor.shape == (batch_size, expected_channels, height, width)

    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()