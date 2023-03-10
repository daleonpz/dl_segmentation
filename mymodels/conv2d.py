import logging
import torch

logger = logging.getLogger(__name__)

class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel=0, bias=True,
                    up=False, down=False, bilinear=False, pooling=False):
        """
        Args:
            in_channels:        Number of input channels for this block.
            out_channels:       Number of output channels for this block.
            kernel:             Kernel size for the first convolution.
            bias:               Bias parameter for the convolution.
            up/down:            Whether the first convolution is in up- or down-sampling mode.
        """

        super().__init__()
        assert not (up and down), 'up and down cannot be both True'
        assert not (kernel and (up or down)), 'Cannot use kernel with up/down sampling'
        assert (kernel or up or down), 'Must use kernel or up or down sampling'

        ### START CODE HERE ### (approx. 23 lines)
        if kernel > 0:
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel, padding=kernel//2, bias=bias)
        elif  down==True:
            if pooling==True:
                self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2)
                    )
            else:
                self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        elif up==True:
            if bilinear==True:
                self.conv = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
                    )
            else:
                self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=bias)

        else:
            raise ValueError('Invalid combination of parameters')
#         else:
#             padding = (kernel-1)//2
#             self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=1, padding=padding, bias=bias)
        ### END CODE HERE ##
        
        logger.debug(self.conv)
        logger.debug('----------------')

    def forward(self, x):
        return self.conv(x)

