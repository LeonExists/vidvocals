"""ConvNormActivation as in torchvision.ops.misc (BSD-3-Clause, (c) Soumith Chintala 2016).

Vendored so that EfficientAT runs without torchvision.
"""
from typing import Callable, Optional, Union, Tuple

import torch


class ConvNormActivation(torch.nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, ...]] = 3,
                 stride: Union[int, Tuple[int, ...]] = 1, padding: Optional[Union[int, Tuple[int, ...], str]] = None,
                 groups: int = 1, norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 dilation: Union[int, Tuple[int, ...]] = 1, inplace: Optional[bool] = True,
                 bias: Optional[bool] = None, conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d) -> None:
        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                n = len(kernel_size) if isinstance(kernel_size, (list, tuple)) else len(dilation)
                ks = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * n
                dl = dilation if isinstance(dilation, (list, tuple)) else (dilation,) * n
                padding = tuple((ks[i] - 1) // 2 * dl[i] for i in range(n))
        if bias is None:
            bias = norm_layer is None
        layers = [conv_layer(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation,
                             groups=groups, bias=bias)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels
