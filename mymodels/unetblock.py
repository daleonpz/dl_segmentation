import torch

import conv2d as mm 

class UNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel=0, up=False, down=False,
                         dropout=0, eps=1e-5, bilinear=False, pooling=False):
        """
        Args:
            in_channels:        Number of input channels for this block.
            out_channels:       Number of output channels for this block.
            kernel:             Kernel size for the first convolution.
            up/down:            Whether the first convolution is in up- or down-sampling mode.
            dropout:            Dropout probability for dropout before the second conv.
        """
        super().__init__()
        ### START CODE HERE ### (approx. 12 lines)
        self.batch_norm = torch.nn.BatchNorm2d(num_features=in_channels, eps=eps)
        self.activation = torch.nn.ReLU()

        if kernel == 0:
            self.conv1 = mm.Conv2d(in_channels, out_channels, up=up, down=down, kernel=kernel, bilinear=bilinear, pooling=pooling)
        else:
            self.conv1  = torch.nn.Sequential(
                    torch.nn.Dropout(dropout),
                    mm.Conv2d(in_channels, out_channels, kernel=3)
                )

        ### END CODE HERE ###

    def forward(self, x):
        residual = x
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.conv1(x)
        print(f'x shape {x.shape}')
        print(f'residual shape {residual.shape}')
        if x.shape != residual.shape:
            return x
#             residual = torch.nn.functional.interpolate(residual, size=x.shape[2:], mode='bilinear', align_corners=True)
        else:
            return x + residual


