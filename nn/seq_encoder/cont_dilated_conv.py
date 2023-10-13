from typing import Tuple

import math

import torch 
import torch.nn as nn
import torch.nn.functional as F

class Kernel(nn.Module):
    """MLP Kernel, takes x of shape (*, in_channels), returns kernel values of shape (*, in_channels, out_channels)."""

    def __init__(
        self,
        hidden1: int,
        hidden2: int,
        hidden3: int,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """Initialize Kernel network.

        args:
            hidden1 - 1st hidden layer size
            hidden2 - 2nd hidden layer size
            hidden3 - 3rd hidden layer size
            in_channels - number of input channels
            out_channels - number of output channels
        """
        super().__init__()
        self.args = [hidden1, hidden2, hidden3, in_channels, out_channels]

        self.layer_1 = nn.Linear(in_channels, hidden1)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(hidden1, hidden2)
        self.relu_2 = nn.ReLU()
        self.layer_3 = nn.Linear(hidden2, hidden3)
        self.relu_3 = nn.ReLU()
        self.layer_4 = nn.Linear(hidden3, in_channels * out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def recreate(self, in_channels: int) -> nn.Module:
        """Copy kernel network.

        args:
            in_channels - number of input channels in a copied kernel

        returns:
            identical kernel network with a new number of input channels
        """
        args = self.args.copy()
        args[3] = in_channels
        return type(self)(*args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwsrd pass of a network.

        args:
            x - input tensor

        returns:
            out - output tensor
        """
        shape = list(x.shape)[:-1]
        shape += [self.in_channels, self.out_channels]
        x = self.relu_1(self.layer_1(x))
        x = self.relu_2(self.layer_2(x))
        x = self.relu_3(self.layer_3(x))
        out = self.layer_4(x)
        out = out.reshape(*shape)

        return out

class ContConv1d(nn.Module):
    """Continuous convolution layer for true events."""

    def __init__(
        self,
        kernel: nn.Module,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        include_zero_lag: bool = False,
    ) -> None:
        """Initialize Continuous convolution layer.

        args:
            kernel - torch.nn.Module, Kernel neural net that takes (*,1) as input and returns (*, in_channles, out_channels) as output
            kernel_size - int, convolution layer kernel size
            in_channels - int, features input size
            out_channles - int, output size
            dilation - int, convolutional layer dilation (default = 1)
            include_zero_lag - bool, indicates if the model should use current time step features for prediction
            skip_connection - bool, indicates if the model should add skip connection in the end, in_channels == out_channels
        """
        super().__init__()
        assert dilation >= 1
        assert in_channels >= 1
        assert out_channels >= 1

        self.kernel = kernel
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.include_zero_lag = include_zero_lag
        self.skip_connection = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )

        self.leaky_relu = nn.LeakyReLU(0.1)

        self.position_vec = torch.tensor(
            [
                math.pow(10000.0, 2.0 * (i // 2) / self.in_channels)
                for i in range(self.in_channels)
            ]
        )

        self.norm = nn.BatchNorm1d(out_channels)

    def __temporal_enc(self, time: torch.Tensor) -> torch.Tensor:
        """Temporal encoding of event sequences.

        args:
            time - true event times

        returns:
            result - encoded times tensor
        """
        result = time.unsqueeze(-1) / self.position_vec.to(time.device)
        result[..., 0::2] = torch.sin(result[..., 0::2])
        result[..., 1::2] = torch.cos(result[..., 1::2])
        return result

    @staticmethod
    def __conv_matrix_constructor(
        times: torch.Tensor,
        features: torch.Tensor,
        non_pad_mask: torch.Tensor,
        kernel_size: int,
        dilation: int,
        include_zero_lag: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns delta_times t_i - t_j, where t_j are true events and the number of delta_times per row is kernel_size.

        args:
            times - torch.Tensor of shape = (bs, max_len), Tensor of all times
            features - torch.Tensor of shape = (bs,max_len, in_channels), input tensor
            non_pad_mask - torch.Tensor of shape = (bs, max_len),  indicates non_pad timestamps
            kernel_size - int, covolution kernel size
            dilation - int, convolution dilation
            include_zero_lag: bool, indicates if we should use zero-lag timestamp

        returns:
            delta_times - torch.Tensor of shape = (bs, kernel_size, max_len) with delta times value between current time and kernel_size true times before it
            pre_conv_features - torch.Tensor of shape = (bs, kernel_size, max_len, in_channels) with corresponding input features of timestamps in delta_times
            dt_mask - torch.Tensor of shape = (bs, kernel_size, max_len), bool tensor that indicates delta_times true values
        """
        # parameters
        padding = (
            (kernel_size - 1) * dilation if include_zero_lag else kernel_size * dilation
        )
        kernel = torch.eye(kernel_size).unsqueeze(1).to(times.device)
        in_channels = features.shape[2]

        # convolutions
        pre_conv_times = F.conv1d(
            times.unsqueeze(1), kernel, padding=padding, dilation=dilation
        )
        pre_conv_features = F.conv1d(
            features.transpose(1, 2),
            kernel.repeat(in_channels, 1, 1),
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        dt_mask = (
            F.conv1d(
                non_pad_mask.float().unsqueeze(1),
                kernel.float(),
                padding=padding,
                dilation=dilation,
            )
            .long()
            .bool()
        )

        # deleting extra values
        pre_conv_times = pre_conv_times[
            :, :, : -(padding + dilation * (1 - int(include_zero_lag)))
        ]
        pre_conv_features = pre_conv_features[
            :, :, : -(padding + dilation * (1 - int(include_zero_lag)))
        ]
        dt_mask = dt_mask[
            :, :, : -(padding + dilation * (1 - int(include_zero_lag)))
        ] * non_pad_mask.unsqueeze(1)

        # updating shape
        bs, L, dim = features.shape
        pre_conv_features = pre_conv_features.reshape(bs, dim, kernel_size, L)

        # computing delte_time and deleting masked values
        delta_times = times.unsqueeze(1) - pre_conv_times
        delta_times[~dt_mask] = 0
        pre_conv_features = pre_conv_features.permute(0, 2, 3, 1)
        pre_conv_features[~dt_mask, :] = 0

        return delta_times, pre_conv_features, dt_mask

    def forward(self, times, features, non_pad_mask):
        """Neural net layer forward pass.

        args:
            times - torch.Tensor, shape = (bs, L), event times
            features - torch.Tensor, shape = (bs, L, in_channels), event features
            non_pad_mask - torch.Tensor, shape = (bs,L), mask that indicates non pad values

        returns:
            out - torch.Tensor, shape = (bs, L, out_channels)
        """
        delta_times, features_kern, dt_mask = self.__conv_matrix_constructor(
            times,
            features,
            non_pad_mask,
            self.kernel_size,
            self.dilation,
            self.include_zero_lag,
        )

        kernel_values = self.kernel(self.__temporal_enc(delta_times))
        kernel_values[~dt_mask, ...] = 0

        out = features_kern.unsqueeze(-1) * kernel_values

        out = out.sum(dim=(1, 3))

        out = out + self.skip_connection(features.transpose(1, 2)).transpose(1, 2)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out