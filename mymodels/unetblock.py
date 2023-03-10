import logging
import torch

import mymodels.conv2d as mm 

logger = logging.getLogger(__name__)

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
            torch.nn.BatchNorm2d(in_channels, eps=eps),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout2d(dropout),
            mm.Conv2d(in_channels, out_channels, kernel=kernel, up=up, down=down, bilinear=bilinear, pooling=pooling),
        )

#         self.conv = torch.nn.Sequential(
#             torch.nn.BatchNorm2d(in_channels),
#             torch.nn.ReLU(inplace=True),
#             mm.Conv2d(in_channels, out_channels, kernel=kernel, up=up, down=down, bilinear=bilinear, pooling=pooling),
#             torch.nn.BatchNorm2d(out_channels),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Dropout(dropout),
#             mm.Conv2d(out_channels, out_channels, kernel=3)
#         )
 
        ### END CODE HERE ###


    def forward(self, x):
        ## START CODE HERE ## (approx. 6 line)
        residual = x
        x = self.conv(x)
        logger.debug("UNetBlock: x.shape: %s, residual.shape: %s", x.shape, residual.shape)
        logger.debug("UNetBlock: x.shape[2:]: %s, residual.shape[2:]: %s", x.shape[2:], residual.shape[2:])

        if residual.shape[2:] == x.shape[2:]:
            return torch.cat((x, residual), dim=1)
        ### END CODE HERE ###
        return x 
