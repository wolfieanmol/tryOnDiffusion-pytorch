import math
import numbers
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F


class Utils:
    @staticmethod
    def batch_norm_groups(ni):
        return min(32, ni // 4)

    @staticmethod
    def saved(m, blk, save_skip_feat=True, save_cross_feat=False):
        m_ = m.forward

        @wraps(m.forward)
        def _f(*args, **kwargs):
            res = m_(*args, **kwargs)
            if save_skip_feat:
                blk.saved.append(res)
            if save_cross_feat:
                blk.saved_block_features.append(res)
            return res

        m.forward = _f
        return m


class GaussianSmoothing(nn.Module):
    """
    Source: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10?u=tanay_agrawal
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data. Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, conv_dim):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * conv_dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * conv_dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if conv_dim == 1:
            self.conv = F.conv1d
        elif conv_dim == 2:
            self.conv = F.conv2d
        elif conv_dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError("Only 1, 2 and 3 dimensions are supported. Received {}.".format(conv_dim))

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
