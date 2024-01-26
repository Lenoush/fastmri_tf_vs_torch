import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv1d(conv_layer):
    def _conv1d_fun(inputs):
        conv = conv_layer(inputs[:, None, :])
        conv = conv[:, 0, :]
        return conv
    return _conv1d_fun


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction_factor=4, dense=False, activation='relu'):
        super(ChannelAttentionBlock, self).__init__()
        self.reduction_factor = reduction_factor
        self.dense = dense
        self.ga_pooling = nn.AdaptiveAvgPool2d(1)
        self.activation_str = activation
        if self.activation_str == 'lrelu':
            self.activation = nn.LeakyReLU(0.1)
        elif self.activation_str == 'prelu':
            self.activation = nn.PReLU()
        else:
            self.activation = nn.ReLU()

        
    def build(self, input_shape):
        n_channels = input_shape[-1]
        n_reduced = n_channels // self.reduction_factor
        
