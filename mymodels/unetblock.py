import torch

import mymodels.conv2d as mm 

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
        self.conv = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
            mm.Conv2d(in_channels, out_channels, up=up, down=down, bilinear=bilinear, pooling=pooling),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            mm.Conv2d(out_channels, out_channels, kernel=3)
        )
 
        ### END CODE HERE ###
    def forward(self, x):
        ## START CODE HERE ## (approx. 6 line)
        residual = x
        x = self.conv(x)
        print(f'x shape {x.shape}')
        print(f'residual shape {residual.shape}')
        if residual.shape == x.shape:
            return x + residual
        ### END CODE HERE ###
        return x 
