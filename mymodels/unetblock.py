import torch
from mymodels.conv2d import Conv2d

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
        ### START CODE HERE ### (approx. 6 lines)
        self.conv1 = Conv2d(in_channels, 64, down=True, pooling=True)
#         self.dropout = torch.nn.Dropout(dropout)
#         self.conv2 = Conv2d(64, 128, down=True, pooling=True)
#         self.conv3 = Conv2d(128, 256, down=True, pooling=True)
#         self.conv4 = Conv2d(256, 512, down=True, pooling=True)
#         self.conv5 = Conv2d(512, 1024, down=True, pooling=True)

        
        ### END CODE HERE ###

    def forward(self, x):
        return x


