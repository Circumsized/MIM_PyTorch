import math

import torch
import torch.nn as nn
import torch.utils.checkpoint


def sequence_to_channels_last(frames):
    time_major = frames.transpose(0, 1).contiguous()
    shape = time_major.shape
    return (
        time_major.reshape(-1, *shape[2:])
        .contiguous(memory_format=torch.channels_last)
        .view(shape)
    )


def _tf_glorot_uniform_(tensor):
    """Initialize a channels-first weight the way TF's default Glorot init
    initializes the equivalent NHWC tensor.

    The official MIM creates ``c_t_weight``/``oc_weight`` via
    ``tf.get_variable`` without an initializer, so TF applies its default
    ``glorot_uniform_initializer``. That computes fan_in/fan_out from the
    last two dims and folds all leading dims into the receptive field: for
    NHWC ``[H, W, C]`` it yields ``fan_in = H*W`` and ``fan_out = C*H``.
    ``tensor`` here is the channels-first ``[C, H, W]`` mirror.
    """
    c, h, w = tensor.shape[0], tensor.shape[1], tensor.shape[2]
    fan_in = h * w
    fan_out = c * h
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        tensor.uniform_(-limit, limit)


def _validate_conv(kernel_size, in_shape):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd int for same-padding convs, "
            f"got {kernel_size}"
        )
    if not (isinstance(in_shape, (list, tuple)) and len(in_shape) == 4):
        raise ValueError(
            f"in_shape must be [batch, channel, height, width], got {in_shape}"
        )
    if in_shape[2] < 1 or in_shape[3] < 1:
        raise ValueError(f"in_shape height/width must be >= 1, got {tuple(in_shape)}")


def _validate_cell_params(in_channel, num_hidden, kernel_size, in_shape):
    if in_channel < 1:
        raise ValueError(f"in_channel must be >= 1, got {in_channel}")
    if num_hidden < 1:
        raise ValueError(f"num_hidden must be >= 1, got {num_hidden}")
    _validate_conv(kernel_size, in_shape)


class TensorLayerNorm(nn.Module):
    """Layer normalization over the [C, H, W] dims of a conv feature map.

    Statistics are always computed in float32 regardless of input dtype so
    the squared-sum inside ``var`` cannot overflow/underflow when running
    under fp16 autocast; the result is cast back to the input dtype.
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        orig_dtype = x.dtype
        x32 = x if orig_dtype == torch.float32 else x.float()
        mean = x32.mean(dim=[1, 2, 3], keepdim=True)
        var = x32.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        x_norm = (x32 - mean) * torch.rsqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta
        return out.to(orig_dtype) if orig_dtype != torch.float32 else out


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(
        self,
        in_channel,
        num_hidden,
        kernel_size,
        in_shape,
        bias=True,
        forget_bias=1.0,
        tln=False,
    ):
        super().__init__()

        _validate_cell_params(in_channel, num_hidden, kernel_size, in_shape)
        self.in_channel = in_channel
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        self.height = in_shape[2]
        self.width = in_shape[3]
        self.padding = kernel_size // 2
        self.bias = bias
        self.forget_bias = forget_bias
        self.layer_norm = tln

        self.t_cc = nn.Conv2d(
            num_hidden,
            num_hidden * 4,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            bias=bias,
        )
        self.s_cc = nn.Conv2d(
            num_hidden,
            num_hidden * 4,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            bias=bias,
        )
        self.x_cc = nn.Conv2d(
            in_channel,
            num_hidden * 4,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            bias=bias,
        )
        self.last = nn.Conv2d(
            num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=bias
        )

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
        return torch.zeros(
            batch_size,
            self.num_hidden,
            self.height,
            self.width,
            dtype=self.t_cc.weight.dtype,
            device=device,
        )

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
    def __init__(
        self,
        in_channel,
        num_hidden,
        kernel_size,
        in_shape,
        stride=1,
        bias=True,
        forget_bias=1.0,
        tln=False,
    ):
        super().__init__()

        _validate_cell_params(in_channel, num_hidden, kernel_size, in_shape)
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

        self.conv_h = nn.Conv2d(
            num_hidden,
            num_hidden * 4,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )
        self.conv_x = nn.Conv2d(
            in_channel,
            num_hidden * 4,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )

        if tln:
            self.tln_h = TensorLayerNorm(num_hidden * 4)
            self.tln_x = TensorLayerNorm(num_hidden * 4)

        self.ct_weight = nn.Parameter(
            torch.empty(num_hidden * 2, self.height, self.width)
        )
        self.oc_weight = nn.Parameter(torch.empty(num_hidden, self.height, self.width))

        self._init_weights()

    def _init_weights(self):
        # MIMBlock.MIMS uses Xavier for the convs (official ``w_initializer``)
        # and TF default Glorot for the spatial modulation weights.
        for conv in [self.conv_h, self.conv_x]:
            fan_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
            fan_out = conv.out_channels * conv.kernel_size[0] * conv.kernel_size[1]
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            nn.init.uniform_(conv.weight, -bound, bound)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        _tf_glorot_uniform_(self.ct_weight)
        _tf_glorot_uniform_(self.oc_weight)

    def init_state(self, batch_size, device):
        return torch.zeros(
            batch_size,
            self.num_hidden,
            self.height,
            self.width,
            dtype=self.conv_h.weight.dtype,
            device=device,
        )

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

        # Equivalent to ``c_t.repeat(1,2,1,1) * ct_weight`` + split, but
        # chunking the weight first avoids allocating the [B, 2h, H, W]
        # intermediate tensor at every timestep.
        w_i, w_f = self.ct_weight.chunk(2, dim=0)
        i_c = c_t * w_i
        f_c = c_t * w_f

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
    def __init__(
        self,
        in_channel,
        num_hidden,
        kernel_size,
        in_shape,
        stride=1,
        bias=True,
        forget_bias=1.0,
        tln=False,
    ):
        super().__init__()
        _validate_cell_params(in_channel, num_hidden, kernel_size, in_shape)
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

        self.t_cc = nn.Conv2d(
            num_hidden,
            num_hidden * 3,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )
        self.s_cc = nn.Conv2d(
            num_hidden,
            num_hidden * 4,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )
        self.x_cc = nn.Conv2d(
            in_channel,
            num_hidden * 4,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )

        if tln:
            self.tln_t = TensorLayerNorm(num_hidden * 3)
            self.tln_s = TensorLayerNorm(num_hidden * 4)
            self.tln_x = TensorLayerNorm(num_hidden * 4)

        # The diff input has `num_hidden` channels (output of the diff MIMN of
        # this layer), NOT `in_channel` (channels of the main-branch input x).
        # The original TF implementation builds MIMS(num_hidden, num_hidden);
        # passing `in_channel` here breaks whenever adjacent hidden dims
        # differ.
        self.mims = MIMS(
            num_hidden,
            num_hidden,
            kernel_size,
            in_shape,
            stride,
            bias,
            forget_bias,
            tln,
        )
        self.last = nn.Conv2d(num_hidden * 2, num_hidden, 1, 1, padding=0, bias=bias)

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
        return torch.zeros(
            batch_size,
            self.num_hidden,
            self.height,
            self.width,
            dtype=self.t_cc.weight.dtype,
            device=device,
        )

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
    def __init__(
        self,
        in_channel,
        num_hidden,
        kernel_size,
        in_shape,
        stride=1,
        bias=True,
        forget_bias=1.0,
        tln=False,
    ):
        super().__init__()
        _validate_cell_params(in_channel, num_hidden, kernel_size, in_shape)
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

        self.conv_h = nn.Conv2d(
            num_hidden,
            num_hidden * 4,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )
        self.conv_x = nn.Conv2d(
            in_channel,
            num_hidden * 4,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )

        if tln:
            self.tln_h = TensorLayerNorm(num_hidden * 4)
            self.tln_x = TensorLayerNorm(num_hidden * 4)

        self.ct_weight = nn.Parameter(
            torch.empty(num_hidden * 2, self.height, self.width)
        )
        self.oc_weight = nn.Parameter(torch.empty(num_hidden, self.height, self.width))

        self._init_weights()

    def _init_weights(self):
        # MIMN (diff branch) initializes its convs with the official
        # ``initializer=0.001`` (±0.001 uniform), NOT Xavier. Its spatial
        # modulation weights still use TF's default Glorot (get_variable).
        for conv in [self.conv_h, self.conv_x]:
            nn.init.uniform_(conv.weight, -0.001, 0.001)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        _tf_glorot_uniform_(self.ct_weight)
        _tf_glorot_uniform_(self.oc_weight)

    def init_state(self, batch_size, device):
        return torch.zeros(
            batch_size,
            self.num_hidden,
            self.height,
            self.width,
            dtype=self.conv_h.weight.dtype,
            device=device,
        )

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

        # Equivalent to ``c_t.repeat(1,2,1,1) * ct_weight`` + split, but
        # chunking the weight first avoids allocating the [B, 2h, H, W]
        # intermediate tensor at every timestep.
        w_i, w_f = self.ct_weight.chunk(2, dim=0)
        i_c = c_t * w_i
        f_c = c_t * w_f

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
    def __init__(
        self,
        input_dims,
        out_dims,
        in_shape,
        hidden_dim=None,
        kernel_size=3,
        stride=1,
        total_length=10,
        input_length=5,
        forget_bias=1.0,
        tln=True,
    ):
        """
        Memory In Memory Networks for spatiotemporal prediction.

        Args:
            input_dims (int): number of input channels
            out_dims (int): number of output channels
            in_shape (list): input shape [batch, channel, height, width]
            hidden_dim (list): hidden layer channels; must be uniform across
                layers because the spatio-temporal memory `m` is shared
                between layers. Default [64, 64, 64, 64]
            kernel_size (int): convolution kernel size
            stride (int): convolution stride
            total_length (int): total number of frames in sequence
            input_length (int): number of input frames
            forget_bias (float): bias added to forget gates for training stability
            tln (bool): whether to apply tensor layer normalization

        Set ``model.gradient_checkpointing = True`` (or use ``--grad_ckpt``
        in train.py) to trade ~1 extra forward pass per step for O(1) instead
        of O(T) activation memory across the temporal loop.
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = [64, 64, 64, 64]
        if not isinstance(hidden_dim, (list, tuple)) or len(hidden_dim) == 0:
            raise ValueError(f"hidden_dim must be a non-empty list, got {hidden_dim!r}")
        if any(d < 1 for d in hidden_dim):
            raise ValueError(f"hidden_dim entries must be >= 1, got {hidden_dim}")
        if len(set(hidden_dim)) != 1:
            raise ValueError(
                "MIM requires uniform hidden_dim across layers because the "
                f"spatio-temporal memory m is shared between layers, got {hidden_dim}"
            )
        if input_dims < 1:
            raise ValueError(f"input_dims must be >= 1, got {input_dims}")
        if out_dims < 1:
            raise ValueError(f"out_dims must be >= 1, got {out_dims}")
        if not isinstance(forget_bias, (int, float)):
            raise ValueError(
                f"forget_bias must be numeric, got {type(forget_bias).__name__}"
            )
        _validate_conv(kernel_size, in_shape)
        if total_length <= input_length:
            raise ValueError(
                f"total_length ({total_length}) must be greater than "
                f"input_length ({input_length})"
            )
        if input_length < 1:
            raise ValueError(f"input_length must be >= 1, got {input_length}")
        if stride != 1:
            raise ValueError(
                f"MIM requires stride=1 (the spatial ct_weight/oc_weight "
                f"modulation is same-shape), got {stride}"
            )

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
        self.gradient_checkpointing = False
        self.channels_last = False
        # st_memory + n hidden + n cell + (n-1) hidden_diff + (n-1) cell_diff
        # + (n-1) convlstm_c — precomputed so the temporal loop avoids any
        # per-forward allocation bookkeeping.
        self._num_states = 5 * self.num_layers - 2

        self.stlstm_layer = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.stlstm_layer.append(
                    SpatioTemporalLSTMCell(
                        input_dims,
                        hidden_dim[i],
                        kernel_size=kernel_size,
                        in_shape=in_shape,
                        forget_bias=forget_bias,
                        tln=tln,
                    )
                )
            else:
                self.stlstm_layer.append(
                    MIMBlock(
                        hidden_dim[i - 1],
                        hidden_dim[i],
                        kernel_size=kernel_size,
                        stride=stride,
                        in_shape=in_shape,
                        forget_bias=forget_bias,
                        tln=tln,
                    )
                )

        self.stlstm_layer_diff = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.stlstm_layer_diff.append(
                MIMN(
                    hidden_dim[i],
                    hidden_dim[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    in_shape=in_shape,
                    forget_bias=forget_bias,
                    tln=tln,
                )
            )

        self.last = nn.Conv2d(hidden_dim[-1], out_dims, 1, 1, 0)
        bound = math.sqrt(6.0 / (hidden_dim[-1] + out_dims))
        nn.init.uniform_(self.last.weight, -bound, bound)
        nn.init.zeros_(self.last.bias)

    def _init_states(self, batch_size, device):
        ch = self.hidden_dim[0]
        dtype = next(self.parameters()).dtype
        states = tuple(
            torch.zeros(
                batch_size, ch, self.height, self.width, dtype=dtype, device=device
            )
            for _ in range(self._num_states)
        )
        if self.channels_last:
            states = tuple(s.to(memory_format=torch.channels_last) for s in states)
        return states

    def _rnn_step(self, ts, frames, ss_bool, x_gen, *states):
        """One timestep over all layers.

        `states` layout (all zero-initialized tensors, 5n-2 entries):
            (st_memory, *hidden_state, *cell_state,
             *hidden_state_diff, *cell_state_diff, *convlstm_c)
        Returns (x_out, *new_states).
        """
        n = self.num_layers
        st_memory = states[0]
        hidden_state = list(states[1 : 1 + n])
        cell_state = list(states[1 + n : 1 + 2 * n])
        hidden_state_diff = list(states[1 + 2 * n : 3 * n])
        cell_state_diff = list(states[3 * n : 4 * n - 1])
        convlstm_c = list(states[4 * n - 1 :])

        if ts < self.input_length:
            x_gen = frames[ts]
        else:
            ss = ss_bool[:, ts - self.input_length].unsqueeze(1)
            # torch.where (not ``ss * frames + (1-ss) * x_gen``): under IEEE
            # semantics ``0 * NaN == NaN``, so a masked frame carrying NaN
            # would poison x_gen even where ss == 0. torch.where selects
            # element-wise and never reads the masked value.
            x_gen = torch.where(ss.bool(), frames[ts], x_gen)

        preh = hidden_state[0]
        hidden_state[0], cell_state[0], st_memory = self.stlstm_layer[0](
            x_gen, hidden_state[0], cell_state[0], st_memory
        )

        for i in range(1, n):
            if ts > 0:
                diff_in = hidden_state[0] - preh if i == 1 else hidden_state_diff[i - 2]
                hidden_state_diff[i - 1], cell_state_diff[i - 1] = (
                    self.stlstm_layer_diff[i - 1](
                        diff_in, hidden_state_diff[i - 1], cell_state_diff[i - 1]
                    )
                )
            # NOTE: the original TF implementation executed a diff-layer call
            # at ts == 0 only to create TF1.x variable scopes; its outputs were
            # discarded. PyTorch eager mode needs no scope warm-up, so the
            # call is omitted here: the diff state stays zero at ts == 0,
            # numerically identical to the TF behaviour.

            preh = hidden_state[i]
            hidden_state[i], cell_state[i], st_memory, convlstm_c[i - 1] = (
                self.stlstm_layer[i](
                    hidden_state[i - 1],
                    hidden_state_diff[i - 1],
                    hidden_state[i],
                    cell_state[i],
                    st_memory,
                    convlstm_c[i - 1],
                )
            )

        x_out = self.last(hidden_state[-1])
        return (
            x_out,
            st_memory,
            *hidden_state,
            *cell_state,
            *hidden_state_diff,
            *cell_state_diff,
            *convlstm_c,
        )

    def forward(self, frames, ss_bool=None):
        """
        Args:
            frames: input tensor [batch, total_length, channel, height, width]
            ss_bool: scheduled sampling tensor
                [batch, total_length - input_length - 1, height, width].
                If None, uses generated frames (no teacher forcing).

        Returns:
            gen_imgs: predicted frames [batch, total_length-1, channel, height, width]
        """
        # ValueError (not assert): input validation must survive ``python -O``.
        if (
            frames.dim() != 5
            or frames.shape[1] < self.total_length
            or frames.shape[2] != self.input_dims
        ):
            raise ValueError(
                f"expect [B,>={self.total_length},{self.input_dims},H,W], "
                f"got {tuple(frames.shape)}"
            )
        if frames.shape[3] != self.height or frames.shape[4] != self.width:
            raise ValueError(
                f"spatial size mismatch: model built for "
                f"({self.height}, {self.width}), got "
                f"({frames.shape[3]}, {frames.shape[4]})"
            )
        batch_size = frames.shape[0]
        device = frames.device
        frames = (
            sequence_to_channels_last(frames)
            if self.channels_last
            else frames.transpose(0, 1)
        )

        expected_ss = (
            batch_size,
            self.total_length - self.input_length - 1,
            self.height,
            self.width,
        )
        if ss_bool is None:
            ss_bool = torch.zeros(*expected_ss, device=device, dtype=frames.dtype)
        else:
            if tuple(ss_bool.shape) != expected_ss:
                raise ValueError(
                    f"ss_bool shape {tuple(ss_bool.shape)} does not match "
                    f"expected {expected_ss}"
                )
            if ss_bool.device.type != device.type:
                ss_bool = ss_bool.to(device=device, dtype=frames.dtype)
            elif ss_bool.dtype != frames.dtype:
                ss_bool = ss_bool.to(dtype=frames.dtype)

        # Zero-initialize all recurrent states up front (tensors instead of
        # None) so the per-step function has a uniform tensor signature,
        # which is required for gradient checkpointing. Semantically identical
        # to the previous lazy None -> zeros initialization.
        states = self._init_states(batch_size, device)

        use_ckpt = (
            self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        )

        gen_imgs = []
        x_gen = frames[0]
        for ts in range(self.total_length - 1):
            if use_ckpt:
                step_out = torch.utils.checkpoint.checkpoint(
                    self._rnn_step,
                    ts,
                    frames,
                    ss_bool,
                    x_gen,
                    *states,
                    use_reentrant=False,
                )
            else:
                step_out = self._rnn_step(ts, frames, ss_bool, x_gen, *states)
            x_gen = step_out[0]
            states = step_out[1:]
            gen_imgs.append(x_gen)

        return torch.stack(gen_imgs, dim=1)
