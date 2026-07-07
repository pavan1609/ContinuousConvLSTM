from torch import nn
import torch
import torch.nn.functional as F


class DeepConvCNN(nn.Module):
    """
    CNN-only baseline: the DeepConvLSTM convolutional front-end with the
    recurrent stage removed. Four temporal conv layers, then global average
    pooling over time, dropout, and a linear classifier. No LSTM.

    Single fixed sampling rate per checkpoint (conv_type == 'standard').
    """

    def __init__(
        self,
        channels,
        classes,
        window_size,
        conv_kernels=64,
        conv_kernel_size=9,
        dropout=0.5,
        standard_padding="valid",
    ):
        super().__init__()
        self.channels = int(channels)
        self.classes = int(classes)
        self.window_size = int(window_size)
        self.conv_kernels = int(conv_kernels)
        self.conv_kernel_size = int(conv_kernel_size)
        self.dropout_p = float(dropout)

        standard_padding = str(standard_padding)
        if standard_padding not in ("valid", "same"):
            raise ValueError(f"standard_padding must be 'valid' or 'same', got: {standard_padding}")
        self.standard_padding = standard_padding
        pad_t = (self.conv_kernel_size // 2) if (self.standard_padding == "same") else 0
        self._std_pad = (pad_t, 0)

        self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1), padding=self._std_pad)
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1), padding=self._std_pad)
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1), padding=self._std_pad)
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1), padding=self._std_pad)

        self.dropout = nn.Dropout(self.dropout_p)
        self.classifier = nn.Linear(self.channels * self.conv_kernels, self.classes)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        if x.ndim != 3:
            raise ValueError(f"DeepConvCNN expected [B,T,C], got shape {tuple(x.shape)}")

        x = x.unsqueeze(1)  # [B, 1, T, C]
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        # [B, K, T, C] -> [T, B, C, K] -> [T, B, C*K]
        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        # global average pool over time -> [B, C*K]  (rate-invariant head)
        x = x.mean(dim=0)
        x = self.dropout(x)
        return self.classifier(x)
