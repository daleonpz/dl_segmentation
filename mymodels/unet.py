import logging
import torch

from mymodels.unetblock import UNetBlock
from mymodels.conv2d    import Conv2d

logger = logging.getLogger(__name__)

class UNet(torch.nn.Module):
    """
    Args:
        resolution:     Image resolution at input/output.
        in_channels:    Number of color channels at input.
        out_channels:   Number of color channels at output.
        kernel_size:    Convolution kernel size.
        base_channels:  Number of channels in first convolution.
        channel_mult:   Per-resolution multipliers for the number of channels.
        num_blocks:     Number of residual blocks per resolution.
        dropout:        Dropout probability of intermediate activations.
        bilinear:       Whether to apply bilinear up-sampling instead of transpose convs
        pooling:        Whether to use max pooling instead of strided convs
    """

    def __init__(self, resolution, in_channels, out_channels, kernel_size=3, 
                 base_channels=32, channel_mult=[1,2,4], num_blocks=1, 
                 dropout=0.10, bilinear=False, pooling=False):

        super(UNet, self).__init__()

        ### START CODE HERE ### (approx. 32 lines)
        self.down_path = torch.nn.ModuleDict()
        self.up_path = torch.nn.ModuleDict()

        self.down_path[f'enc.{resolution}x{resolution}_in_conv'] =  Conv2d(in_channels, base_channels, kernel=kernel_size)

        for i in range(len(channel_mult)):
            self.down_path[f'enc.{resolution // 2**(i+1)}x{resolution // 2**(i+1)}_block{i}'] = UNetBlock(
                    base_channels * channel_mult[i], 
                    base_channels * channel_mult[i+1], 
                    kernel=kernel_size, dropout=dropout, pooling=pooling)
# 
#         for i in range(len(channel_mult)-1):
#             self.up_path[f'dec.{resolution // 2**(i+1)}x{resolution // 2**(i+1)}_block{i}'] = UNetBlock(
#                     base_channels * channel_mult[i+1], 
#                     base_channels * channel_mult[i], 
#                     kernel=kernel_size, dropout=dropout, bilinear=bilinear)
#     
#         self.up_path[f'dec.{resolution}x{resolution}_out_conv'] = UNetBlock(
#                 base_channels * channel_mult[0],
#                 out_channels,
#                 kernel=kernel_size, dropout=dropout, bilinear=bilinear)
# 


        ### END CODE HERE ###
# 
#         for i in range(num_downs - 1):
#             self.down_path.append(unetblock.DownBlock(num_filters, num_filters * 2))
#             num_filters *= 2
#         for i in range(num_downs - 1):
#             self.up_path.append(unetblock.UpBlock(num_filters, num_filters // 2))
#             num_filters //= 2
#         self.up_path.append(unetblock.UpBlock(num_filters, out_channels, use_act=False))
# 
#         num_filters = base_channels
#         self.up_path[f'{num_filters}x{num_filters}_out_conv'] = torch.nn.Sequential(
#             torch.nn.BatchNorm2d(base_channels * channel_mult[-1]), 
#             torch.nn.ReLU(inplace=True),
#             Conv2d(base_channels * channel_mult[-1], out_channels, kernel=3)
#             )

    def forward(self, x):
		### START CODE HERE ### (approx. 4 lines)

		### END CODE HERE ###
#         blocks = []
        for down in self.down_path:
            x = down(x)
#             blocks.append(x)
#         blocks = blocks[:-1]
#         blocks.reverse()
#         for i, up in enumerate(self.up_path):
#             x = up(x, blocks[i])
        return x
