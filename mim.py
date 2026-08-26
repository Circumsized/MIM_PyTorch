import math

import torch
import torch.nn as nn


class TensorLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(TensorLayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        var = x.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, kernel_size, in_shape,
                 bias=True, forget_bias=1.0, tln=False):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.in_channel = in_channel
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        self.batch = in_shape[0]
        self.height = in_shape[2]
        self.width = in_shape[3]
        self.padding = kernel_size // 2
        self.bias = bias
        self.forget_bias = forget_bias
        self.layer_norm = tln

        self.t_cc = nn.Conv2d(num_hidden, num_hidden * 4,
                              kernel_size=kernel_size, stride=1,
                              padding=self.padding, bias=bias)
        self.s_cc = nn.Conv2d(num_hidden, num_hidden * 4,
                              kernel_size=kernel_size, stride=1,
                              padding=self.padding, bias=bias)
        self.x_cc = nn.Conv2d(in_channel, num_hidden * 4,
                              kernel_size=kernel_size, stride=1,
                              padding=self.padding, bias=bias)
        self.last = nn.Conv2d(num_hidden * 2, num_hidden,
                              kernel_size=1, stride=1, padding=0, bias=bias)

        if tln:
            self.tln_t = TensorLayerNorm(num_hidden * 4)
            self.tln_s = TensorLayerNorm(num_hidden * 4)
            self.tln_x = TensorLayerNorm(num_hidden * 4)

        self._init_weights()

    def _init_weights(self):
        for conv in [self.t_cc, self.s_cc, self.x_cc]:
            fan_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
            fan_out = conv.out_channels * conv.kernel_size[0] * conv.kernel_size[1]
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            nn.init.uniform_(conv.weight, -bound, bound)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        fan_in = self.last.in_channels
        fan_out = self.last.out_channels
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.last.weight, -bound, bound)
        if self.last.bias is not None:
            nn.init.zeros_(self.last.bias)

    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.num_hidden, self.height, self.width,
                           dtype=torch.float32, device=device)

    def forward(self, x, h, c, m):
        batch_size = x.shape[0]
        if h is None:
            h = self.init_state(batch_size, x.device)
        if c is None:
            c = self.init_state(batch_size, x.device)
        if m is None:
            m = self.init_state(batch_size, x.device)

        t_cc = self.t_cc(h)
        s_cc = self.s_cc(m)
        x_cc = self.x_cc(x)

        if self.layer_norm:
            t_cc = self.tln_t(t_cc)
            s_cc = self.tln_s(s_cc)
            x_cc = self.tln_x(x_cc)

        i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, dim=1)
        i_t, g_t, f_t, o_t = torch.split(t_cc, self.num_hidden, dim=1)
        i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, dim=1)

        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f = torch.sigmoid(f_x + f_t + self.forget_bias)
        f_ = torch.sigmoid(f_x + f_s + self.forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)
        new_m = f_ * m + i_ * g_
        new_c = f * c + i * g

        cell = torch.cat([new_c, new_m], dim=1)
        cell = self.last(cell)
        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m


class MIMS(nn.Module):
    def __init__(self, in_channel, num_hidden, kernel_size, in_shape,
                 stride=1, bias=True, forget_bias=1.0, tln=False):
        super(MIMS, self).__init__()

        self.in_channel = in_channel
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.height = in_shape[2]
        self.width = in_shape[3]
        self.padding = kernel_size // 2
        self.bias = bias
        self.forget_bias = forget_bias
        self.layer_norm = tln

        self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4,
                                kernel_size=kernel_size, stride=stride,
                                padding=self.padding, bias=bias)
        self.conv_x = nn.Conv2d(in_channel, num_hidden * 4,
                                kernel_size=kernel_size, stride=stride,
                                padding=self.padding, bias=bias)

        if tln:
            self.tln_h = TensorLayerNorm(num_hidden * 4)
            self.tln_x = TensorLayerNorm(num_hidden * 4)

        self.ct_weight = nn.Parameter(
            torch.empty(num_hidden * 2, self.height, self.width))
        self.oc_weight = nn.Parameter(
            torch.empty(num_hidden, self.height, self.width))

        self._init_weights()

    def _init_weights(self):
        for conv in [self.conv_h, self.conv_x]:
            fan_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
            fan_out = conv.out_channels * conv.kernel_size[0] * conv.kernel_size[1]
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            nn.init.uniform_(conv.weight, -bound, bound)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        nn.init.kaiming_normal_(self.ct_weight)
        nn.init.kaiming_normal_(self.oc_weight)

    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.num_hidden, self.height, self.width,
                           dtype=torch.float32, device=device)

    def forward(self, x, h_t, c_t):
        batch_size = h_t.shape[0] if h_t is not None else x.shape[0]
        if h_t is None:
            h_t = self.init_state(batch_size, x.device)
        if c_t is None:
            c_t = self.init_state(batch_size, x.device)

        h_concat = self.conv_h(h_t)
        if self.layer_norm:
            h_concat = self.tln_h(h_concat)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        ct_activation = c_t.repeat(1, 2, 1, 1) * self.ct_weight
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x is not None:
            x_concat = self.conv_x(x)
            if self.layer_norm:
                x_concat = self.tln_x(x_concat)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
            i_ = i_ + i_x
            f_ = f_ + f_x
            g_ = g_ + g_x
            o_ = o_ + o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self.forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = o_ + c_new * self.oc_weight
        h_new = torch.sigmoid(o_c) * torch.tanh(c_new)

        return h_new, c_new


class MIMBlock(nn.Module):
    def __init__(self, in_channel, num_hidden, kernel_size, in_shape,
                 stride=1, bias=True, forget_bias=1.0, tln=False):
        super(MIMBlock, self).__init__()
        self.in_channel = in_channel
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.height = in_shape[2]
        self.width = in_shape[3]
        self.padding = kernel_size // 2
        self.bias = bias
        self.forget_bias = forget_bias
        self.layer_norm = tln

        self.t_cc = nn.Conv2d(num_hidden, num_hidden * 3,
                              kernel_size=kernel_size, stride=stride,
                              padding=self.padding, bias=bias)
        self.s_cc = nn.Conv2d(num_hidden, num_hidden * 4,
                              kernel_size=kernel_size, stride=stride,
                              padding=self.padding, bias=bias)
        self.x_cc = nn.Conv2d(in_channel, num_hidden * 4,
                              kernel_size=kernel_size, stride=stride,
                              padding=self.padding, bias=bias)

        if tln:
            self.tln_t = TensorLayerNorm(num_hidden * 3)
            self.tln_s = TensorLayerNorm(num_hidden * 4)
            self.tln_x = TensorLayerNorm(num_hidden * 4)

        self.mims = MIMS(in_channel, num_hidden, kernel_size, in_shape,
                         stride, bias, forget_bias, tln)
        self.last = nn.Conv2d(num_hidden * 2, num_hidden, 1, 1, padding=0)

        self._init_weights()

    def _init_weights(self):
        for conv in [self.t_cc, self.s_cc, self.x_cc]:
            fan_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
            fan_out = conv.out_channels * conv.kernel_size[0] * conv.kernel_size[1]
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            nn.init.uniform_(conv.weight, -bound, bound)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        bound = math.sqrt(6.0 / (self.last.in_channels + self.last.out_channels))
        nn.init.uniform_(self.last.weight, -bound, bound)
        if self.last.bias is not None:
            nn.init.zeros_(self.last.bias)

    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.num_hidden, self.height, self.width,
                           dtype=torch.float32, device=device)

    def forward(self, x, diff_h, h, c, m, convlstm_c):
        batch_size = x.shape[0]
        if h is None:
            h = self.init_state(batch_size, x.device)
        if c is None:
            c = self.init_state(batch_size, x.device)
        if m is None:
            m = self.init_state(batch_size, x.device)
        if convlstm_c is None:
            convlstm_c = self.init_state(batch_size, x.device)
        if diff_h is None:
            diff_h = torch.zeros_like(h)

        t_cc = self.t_cc(h)
        s_cc = self.s_cc(m)
        x_cc = self.x_cc(x)

        if self.layer_norm:
            t_cc = self.tln_t(t_cc)
            s_cc = self.tln_s(s_cc)
            x_cc = self.tln_x(x_cc)

        i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, dim=1)
        i_t, g_t, o_t = torch.split(t_cc, self.num_hidden, dim=1)
        i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, dim=1)

        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f_ = torch.sigmoid(f_x + f_s + self.forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)
        new_m = f_ * m + i_ * g_

        c, convlstm_c = self.mims(diff_h, c, convlstm_c)

        new_c = c + i * g
        cell = torch.cat([new_c, new_m], dim=1)
        cell = self.last(cell)
        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m, convlstm_c


class MIMN(nn.Module):
    def __init__(self, in_channel, num_hidden, kernel_size, in_shape,
                 stride=1, bias=True, forget_bias=1.0, tln=False):
        super(MIMN, self).__init__()
        self.in_channel = in_channel
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.height = in_shape[2]
        self.width = in_shape[3]
        self.padding = kernel_size // 2
        self.bias = bias
        self.forget_bias = forget_bias
        self.layer_norm = tln

        self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4,
                                kernel_size=kernel_size, stride=1,
                                padding=self.padding, bias=bias)
        self.conv_x = nn.Conv2d(in_channel, num_hidden * 4,
                                kernel_size=kernel_size, stride=1,
                                padding=self.padding, bias=bias)

        if tln:
            self.tln_h = TensorLayerNorm(num_hidden * 4)
            self.tln_x = TensorLayerNorm(num_hidden * 4)

        self.ct_weight = nn.Parameter(
            torch.empty(num_hidden * 2, self.height, self.width))
        self.oc_weight = nn.Parameter(
            torch.empty(num_hidden, self.height, self.width))

        self._init_weights()

    def _init_weights(self):
        for conv in [self.conv_h, self.conv_x]:
            fan_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
            fan_out = conv.out_channels * conv.kernel_size[0] * conv.kernel_size[1]
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            nn.init.uniform_(conv.weight, -bound, bound)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        nn.init.kaiming_normal_(self.ct_weight)
        nn.init.kaiming_normal_(self.oc_weight)

    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.num_hidden, self.height, self.width,
                           dtype=torch.float32, device=device)

    def forward(self, x, h_t, c_t):
        batch_size = h_t.shape[0] if h_t is not None else x.shape[0]
        if h_t is None:
            h_t = self.init_state(batch_size, x.device)
        if c_t is None:
            c_t = self.init_state(batch_size, x.device)

        h_concat = self.conv_h(h_t)
        if self.layer_norm:
            h_concat = self.tln_h(h_concat)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        ct_activation = c_t.repeat(1, 2, 1, 1) * self.ct_weight
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x is not None:
            x_concat = self.conv_x(x)
            if self.layer_norm:
                x_concat = self.tln_x(x_concat)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
            i_ = i_ + i_x
            f_ = f_ + f_x
            g_ = g_ + g_x
            o_ = o_ + o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self.forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = torch.sigmoid(o_ + c_new * self.oc_weight)
        h_new = o_c * torch.tanh(c_new)

        return h_new, c_new


class MIM(nn.Module):
    def __init__(self, input_dims, out_dims, in_shape, hidden_dim=None,
                 kernel_size=3, stride=1, total_length=10, input_length=5,
                 forget_bias=1.0, tln=True):
        """
        Memory In Memory Networks for spatiotemporal prediction.

        Args:
            input_dims (int): number of input channels
            out_dims (int): number of output channels
            in_shape (list): input shape [batch, channel, height, width]
            hidden_dim (list): hidden layer channels, default [64, 64, 64, 64]
            kernel_size (int): convolution kernel size
            stride (int): convolution stride
            total_length (int): total number of frames in sequence
            input_length (int): number of input frames
            forget_bias (float): bias added to forget gates for training stability
            tln (bool): whether to apply tensor layer normalization
        """
        super(MIM, self).__init__()

        if hidden_dim is None:
            hidden_dim = [64, 64, 64, 64]

        self.input_dims = input_dims
        self.out_dims = out_dims
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_layers = len(hidden_dim)
        self.total_length = total_length
        self.input_length = input_length
        self.in_shape = in_shape
        self.height = in_shape[2]
        self.width = in_shape[3]
        self.forget_bias = forget_bias
        self.tln = tln

        self.stlstm_layer = nn.ModuleList()
        for i in range(self.num_layers):
            if i < 1:
                self.stlstm_layer.append(
                    SpatioTemporalLSTMCell(
                        input_dims, hidden_dim[i],
                        kernel_size=kernel_size, in_shape=in_shape,
                        forget_bias=forget_bias, tln=tln))
            else:
                self.stlstm_layer.append(
                    MIMBlock(
                        hidden_dim[i - 1], hidden_dim[i],
                        kernel_size=kernel_size, stride=stride,
                        in_shape=in_shape,
                        forget_bias=forget_bias, tln=tln))

        self.stlstm_layer_diff = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.stlstm_layer_diff.append(
                MIMN(hidden_dim[i], hidden_dim[i + 1],
                     kernel_size=kernel_size, stride=stride,
                     in_shape=in_shape,
                     forget_bias=forget_bias, tln=tln))

        self.last = nn.Conv2d(hidden_dim[-1], out_dims, 1, 1, 0)
        bound = math.sqrt(6.0 / (hidden_dim[-1] + out_dims))
        nn.init.uniform_(self.last.weight, -bound, bound)
        nn.init.zeros_(self.last.bias)

    def forward(self, frames, ss_bool=None):
        """
        Args:
            frames: input tensor [batch, total_length, channel, height, width]
            ss_bool: scheduled sampling boolean tensor
                     [batch, total_length - input_length - 1, height, width]
                     If None, uses generated frames (no teacher forcing).

        Returns:
            gen_imgs: predicted frames [batch, total_length-1, channel, height, width]
        """
        assert frames.dim() == 5 and frames.shape[1] >= self.total_length \
            and frames.shape[2] == self.input_dims, \
            f"expect [B,>={self.total_length},{self.input_dims},H,W], got {tuple(frames.shape)}"
        batch_size = frames.shape[0]
        device = frames.device

        st_memory = None
        cell_state = [None] * self.num_layers
        hidden_state = [None] * self.num_layers
        cell_state_diff = [None] * (self.num_layers - 1)
        hidden_state_diff = [None] * (self.num_layers - 1)
        convlstm_c = [None] * (self.num_layers - 1)

        if ss_bool is None:
            ss_bool = torch.zeros(
                batch_size, self.total_length - self.input_length - 1,
                self.height, self.width, device=device)

        gen_imgs = []
        for ts in range(self.total_length - 1):
            if ts < self.input_length:
                x_gen = frames[:, ts]
            else:
                ss = ss_bool[:, ts - self.input_length].unsqueeze(1)
                x_gen = ss * frames[:, ts] + (1 - ss) * x_gen

            preh = hidden_state[0]
            hidden_state[0], cell_state[0], st_memory = self.stlstm_layer[0](
                x_gen, hidden_state[0], cell_state[0], st_memory)

            for i in range(1, self.num_layers):
                if ts > 0:
                    if i == 1:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = \
                            self.stlstm_layer_diff[i - 1](
                                hidden_state[i - 1] - preh,
                                hidden_state_diff[i - 1],
                                cell_state_diff[i - 1])
                    else:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = \
                            self.stlstm_layer_diff[i - 1](
                                hidden_state_diff[i - 2],
                                hidden_state_diff[i - 1],
                                cell_state_diff[i - 1])
                else:
                    self.stlstm_layer_diff[i - 1](
                        torch.zeros_like(hidden_state[i - 1]), None, None)

                preh = hidden_state[i]
                hidden_state[i], cell_state[i], st_memory, convlstm_c[i - 1] = \
                    self.stlstm_layer[i](
                        hidden_state[i - 1], hidden_state_diff[i - 1],
                        hidden_state[i], cell_state[i], st_memory,
                        convlstm_c[i - 1])

            x_gen = self.last(hidden_state[-1])
            gen_imgs.append(x_gen)

        return torch.stack(gen_imgs, dim=1)
