import torch

class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, bias=True,
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
        assert kernel or up or down, 'Must use kernel or up or down sampling'
        ### START CODE HERE ### (approx. 23 lines)
        if up:
            if pooling:
                kernel, padding = 3, 1
            else:
                kernel, padding = 2, 0

            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=1, padding=padding, bias=bias)
        elif down:
            if bilinear:
                self.conv = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
                    )
            else:
                self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=bias)
        ### END CODE HERE ##
 
    def forward(self, x):
        return self.conv(x)


