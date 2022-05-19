import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from timm.models.layers import trunc_normal_

from op import fused_leaky_relu, upfirdn2d, FusedLeakyReLU

def window_partition(x, window_size):  # x : [BS x H x W x C] => [BS*nW x win x win x C]
    BS, H, W, C = x.shape
    x = x.reshape(BS, H//window_size, window_size, W//window_size, window_size, C)
    x = x.permute(0,1,3,2,4,5).contiguous().reshape(-1, window_size, window_size, C)
    return x
def window_reverse(x, window_size, H, W):  # x : [BS*nW x win x win x C] => [BS x H x W x C]    
    nH = H//window_size
    nW = W//window_size
    BS = x.shape[0] // (nH*nW)
    x = x.reshape(BS, nH, nW, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().reshape(BS, H, W, -1)
    return x
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((out_dim, in_dim)).div_(lr_mul))
        self.lr_mul = lr_mul

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
    def forward(self, x):
        if self.activation:
            x = F.linear(x, self.weight * self.scale)
            x = fused_leaky_relu(x, self.bias*self.lr_mul)
        else:
            x = F.linear(x, self.weight*self.scale, bias=self.bias*self.lr_mul)
        return x
def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k
class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out
class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)
    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)
        return out
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self,
                 embedding_dim,
                 padding_idx,
                 init_size=1024,
                 div_half_dim=False,
                 center_shift=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.div_half_dim = div_half_dim
        self.center_shift = center_shift

        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx, self.div_half_dim)

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

        self.max_positions = int(1e5)

    @staticmethod
    def get_embedding(num_embeddings,
                      embedding_dim,
                      padding_idx=None,
                      div_half_dim=False):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert embedding_dim % 2 == 0, (
            'In this version, we request '
            f'embedding_dim divisible by 2 but got {embedding_dim}')

        # there is a little difference from the original paper.
        half_dim = embedding_dim // 2
        if not div_half_dim:
            emb = np.log(10000) / (half_dim - 1)
        else:
            emb = np.log(1e4) / half_dim
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(
            num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input, **kwargs):
        """Input is expected to be of size [bsz x seqlen].
        Returned tensor is expected to be of size  [bsz x seq_len x emb_dim]
        """
        assert input.dim() == 2 or input.dim(
        ) == 4, 'Input dimension should be 2 (1D) or 4(2D)'

        if input.dim() == 4:
            return self.make_grid2d_like(input, **kwargs)

        b, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embedding if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        positions = self.make_positions(input, self.padding_idx).to(
            self._float_tensor.device)

        return self.weights.index_select(0, positions.view(-1)).view(
            b, seq_len, self.embedding_dim).detach()

    def make_positions(self, input, padding_idx):
        mask = input.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) *
                mask).long() + padding_idx

    def make_grid2d(self, height, width, num_batches=1, center_shift=None):
        h, w = height, width
        # if `center_shift` is not given from the outside, use
        # `self.center_shift`
        if center_shift is None:
            center_shift = self.center_shift

        h_shift = 0
        w_shift = 0
        # center shift to the input grid
        if center_shift is not None:
            # if h/w is even, the left center should be aligned with
            # center shift
            if h % 2 == 0:
                h_left_center = h // 2
                h_shift = center_shift - h_left_center
            else:
                h_center = h // 2 + 1
                h_shift = center_shift - h_center

            if w % 2 == 0:
                w_left_center = w // 2
                w_shift = center_shift - w_left_center
            else:
                w_center = w // 2 + 1
                w_shift = center_shift - w_center

        # Note that the index is started from 1 since zero will be padding idx.
        # axis -- (b, h or w)
        x_axis = torch.arange(1, w + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + w_shift
        y_axis = torch.arange(1, h + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + h_shift

        # emb -- (b, emb_dim, h or w)
        x_emb = self(x_axis).transpose(1, 2)
        y_emb = self(y_axis).transpose(1, 2)

        # make grid for x/y axis
        # Note that repeat will copy data. If use learned emb, expand may be
        # better.
        x_grid = x_emb.unsqueeze(2).repeat(1, 1, h, 1)
        y_grid = y_emb.unsqueeze(3).repeat(1, 1, 1, w)

        # cat grid -- (b, 2 x emb_dim, h, w)
        grid = torch.cat([x_grid, y_grid], dim=1)
        return grid.detach()

    def make_grid2d_like(self, x, center_shift=None):
        """Input tensor with shape of (b, ..., h, w) Return tensor with shape
        of (b, 2 x emb_dim, h, w)
        Note that the positional embedding highly depends on the the function,
        ``make_positions``.
        """
        h, w = x.shape[-2:]
        grid = self.make_grid2d(h, w, x.size(0), center_shift)

        return grid.to(x)
class ToRGB(nn.Module):
    def __init__(self, in_ch, is_upsample=True, blur_kernel=[1,3,3,1]):
        super().__init__()
        self.is_upsample = is_upsample
        self.conv = nn.Conv2d(in_ch, 3, kernel_size=1)
        self.bias = nn.Parameter(torch.zeros((1,3,1,1)))

        if is_upsample:
            self.upsample = Upsample(blur_kernel)
    def forward(self, x, skip=None):
        out = self.conv(x)
        out = out + self.bias
        if skip is not None:
            if self.is_upsample:
                skip = self.upsample(skip)
            out = out + skip
        return out  
class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn((1, channel, size, size)))
    def forward(self, x):
        BS = x.shape[0]
        return self.input.repeat(BS, 1, 1, 1)
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
class CustomAct(nn.Module):
    def __init__(self, act_type):
        super().__init__()
        if act_type == "gelu": self.act_layer = gelu
        elif act_type == "l_relu": self.act_layer = nn.LeakyReLU(0.2)
    def forward(self, x):
        return self.act_layer(x)     
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, act_type="gelu", drop=0.0):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        self.fc_1 = nn.Linear(in_dim, hidden_dim)
        self.act_layer = CustomAct(act_type)
        self.fc_2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc_1(x)
        x = self.act_layer(x)
        x = self.drop(x)
        x = self.fc_2(x)
        x = self.drop(x)
        return x
class AdaIN(nn.Module):
    def __init__(self, in_ch, style_dim): 
        super().__init__()
        self.adapt_layer = EqualLinear(style_dim, in_ch*2)
        self.norm = nn.InstanceNorm1d(in_ch)
    def forward(self, x, style):
        x = x.transpose(-2,-1)  # input x의 shape이 [BS x N x C]일 때 채널은 C임을 명심하자. 따라서 이 코드에서 x가 instance norm을 거치기 위해서는 x의 shape은 [BS x C x N]이 되어야한다. 
        style = self.adapt_layer(style).unsqueeze(-1)  # [BS x style_dim] => [BS x in_ch*2 x 1]
        gamma, beta = style.chunk(2,1)  # [BS x in_ch x 1]
        x = self.norm(x)  # [BS x in_ch x N]
        x = gamma * x + beta
        x = x.transpose(-2, -1)
        return x             
class WindowAttention(nn.Module):  # double attention에서 두개의 qkv에 각각 사용된다. 따라서 기존 swin transformer의 WindowAttention 모듈과 달리 입력이 x가 아닌 qkv이다.
    def __init__(self, dim, window_size, n_heads, qk_scale=None, attn_drop=0.0):
        super().__init__()
        self.window_size = window_size
        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.qk_scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords = coords.flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]  # [2 x win**2 x win**2]
        relative_coords = relative_coords.permute(1,2,0).contiguous()  # [win**2 x win**2 x 2]
        relative_coords[:,:,0] += window_size - 1
        relative_coords[:,:,1] += window_size - 1
        relative_coords[:,:,0] += 2*window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.relative_position_bias_table = nn.Parameter(torch.zeros(((2*window_size-1)**2, n_heads)))
        trunc_normal_(self.relative_position_bias_table, std=0.02)
    def forward(self, qkv, mask=None):  # qkv : [3 x BS x H x W x C//2] forward의 코드에서는 C//2를 C로 사용할 것이지만, 큰 흐름으로 봤을때는 double attention을 사용하므로 C//2가 맞다. DoubleAttention 모듈에서는 2개의 WindowAttention이 사용되는데 하나는 shift가 안된것, 하나는 된것.
        q, k, v = qkv[0], qkv[1], qkv[2]  # [BS x H x W x C//2]
        BS, H, W, C = q.shape
        q = window_partition(q, self.window_size).reshape(-1, self.window_size**2, C)
        k = window_partition(k, self.window_size).reshape(-1, self.window_size**2, C)
        v = window_partition(v, self.window_size).reshape(-1, self.window_size**2, C)
        BS, N, C = q.shape
        q = q.reshape(BS, N, self.n_heads, C//self.n_heads).permute(0,2,1,3)
        k = k.reshape(BS, N, self.n_heads, C//self.n_heads).permute(0,2,1,3)
        v = v.reshape(BS, N, self.n_heads, C//self.n_heads).permute(0,2,1,3)

        attn = q@k.transpose(-2,-1) * self.qk_scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]  # [(win**2)**2 x n_heads]
        relative_position_bias = relative_position_bias.reshape(self.window_size**2, self.window_size**2, self.n_heads)
        relative_position_bias = relative_position_bias.permute(2,0,1).contiguous()  # [n_heads x win**2 x win**2]
        attn = attn + relative_position_bias.unsqueeze(0)  # [BS*nW x n_heads x win**2 x win**2]

        if mask is not None:  # mask : [nW x win**2 x win**2]
            n_windows = mask.shape[0]
            attn = attn.reshape(BS // n_windows, n_windows, self.n_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(BS, self.n_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = attn@v  # [BS x n_heads x N x C//n_heads]
        x = x.transpose(1,2)
        x = x.reshape(BS, N, C)  # 기존의 windowattention과 달리 여기서 projection하지 않는다.
        return x
class StyleSwinBlock(nn.Module):
    def __init__(self, dim, style_dim, window_size, in_res, n_heads, qk_scale=None, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, drop=0.0, mlp_ratio=4.0, act_type="gelu"):
        super().__init__()
        self.in_res = in_res
        self.window_size = window_size
        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.qk_scale = qk_scale or head_dim ** -0.5
        
        self.norm1 = AdaIN(dim, style_dim)
        self.qkv_layer = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn1 = WindowAttention(dim=dim//2, window_size=window_size, n_heads=n_heads//2, qk_scale=qk_scale, attn_drop=attn_drop)
        self.attn2 = WindowAttention(dim=dim//2, window_size=window_size, n_heads=n_heads//2, qk_scale=qk_scale, attn_drop=attn_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        H = W = in_res
        img_mask = torch.zeros((1,H,W,1))
        self.shift_size = window_size // 2
        h_slices = [
            slice(0, -window_size),
            slice(-window_size, -self.shift_size),
            slice(-self.shift_size, None)
        ]
        w_slices = [
            slice(0, -window_size),
            slice(-window_size, -self.shift_size),
            slice(-self.shift_size, None)
        ]
        cnt = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, window_size).reshape(-1, window_size**2)
        attn_mask2 = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask2 = attn_mask2.masked_fill(attn_mask2 !=0, float(-100.0)).masked_fill(attn_mask2==0, float(0.0))
        self.register_buffer("attn_mask1", None)  # 첫번째 attn1 레이어에 들어가는 shift 하지 않은 입력
        self.register_buffer("attn_mask2", attn_mask2)  #  두번째 shift 한 입력

        mlp_hidden_dim = int(mlp_ratio * dim)
        self.norm2 = AdaIN(dim, style_dim)
        self.mlp = MLP(dim, hidden_dim=mlp_hidden_dim, act_type=act_type, drop=drop)
    def forward(self, x, style):  # x : [BS x N x C] 
        H = W = self.in_res
        BS, N, C = x.shape
        assert N == H*W
        shortcut = x
        x = self.norm1(x, style)
        qkv = self.qkv_layer(x).reshape(BS, N, 3, C).permute(2,0,1,3).reshape(3*BS, H, W, C)  # 여기서 reshape을 [3 x BS x H x W x C]로 안하는 이유는 qkv2가 torch.roll 들어가야한다.
        qkv1 = qkv[:,:,:,:C//2]
        qkv2 = qkv[:,:,:,C//2:]
        qkv2 = torch.roll(qkv2, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
        qkv1 = qkv1.reshape(3, BS, H, W, C//2)
        qkv2 = qkv2.reshape(3, BS, H, W, C//2)
        # Double Attention 
        x1 = self.attn1(qkv1, self.attn_mask1)  # [BS*nW x win**2 x C//2]
        x2 = self.attn2(qkv2, self.attn_mask2)  # [BS*nW x win**2 x C//2]

        x1 = window_reverse(x1.reshape(-1, self.window_size, self.window_size, C//2), window_size=self.window_size, H=H, W=W)  # [BS x H x W x C//2]
        x2 = window_reverse(x2.reshape(-1, self.window_size, self.window_size, C//2), window_size=self.window_size, H=H, W=W)  # [BS x H x W x C//2]

        x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(1,2))
        x = torch.cat([x1, x2], dim=-1).reshape(BS, N, C)

        x = self.proj(x)
        x = shortcut + x 
        x = x + self.mlp(self.norm2(x, style))
        return x
class BilinearUpsample(nn.Module):
    def __init__(self, in_res, dim, out_dim=None):
        super().__init__()
        self.in_res = in_res
        self.upsample_layer = nn.Upsample(scale_factor=2, mode="bicubic")
        self.norm = nn.LayerNorm(dim)
        self.reduction = nn.Linear(dim, out_dim)
        self.sin_pos_embed = SinusoidalPositionalEmbedding(embedding_dim=out_dim//2, padding_idx=0, init_size=out_dim//2)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        H = W = self.in_res
        BS, N, C = x.shape
        x = x.reshape(BS, H, W, C).permute(0,3,1,2).contiguous()
        x = self.upsample_layer(x)
        x = x.permute(0,2,3,1).contiguous().reshape(BS, 4*N, C)
        x = self.norm(x)  # [BS x 4N x C]
        x = self.reduction(x)  # [BS x 4N x out_dim]
        x = x.reshape(BS, 2*H, 2*W, -1).permute(0,3,1,2)
        x += self.sin_pos_embed.make_grid2d(2*H, 2*W, BS) * self.alpha
        x = x.permute(0,2,3,1).contiguous().reshape(BS, 2*H * 2*W, -1)
        return x
class StageBlock(nn.Module):
    def __init__(self, dim, style_dim, window_size, in_res, n_heads, depth=2, qk_scale=None, qkv_bias=True, mlp_ratio=4.0, act_type="gelu", drop=0.0, attn_drop=0.0, proj_drop=0.0, upsample=None, out_dim=None):
        super().__init__()
        # Figure 2(c)에 있는 double attention
        self.layers = nn.ModuleList([
            StyleSwinBlock(dim=dim, style_dim=style_dim, window_size=window_size, in_res=in_res, n_heads=n_heads, qk_scale=qk_scale, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, drop=drop, mlp_ratio=mlp_ratio, act_type=act_type)
        for _ in range(depth)])
        if upsample is not None:
            self.upsample = BilinearUpsample(in_res=in_res, dim=dim, out_dim=out_dim)
        else: 
            self.upsample = None
    def forward(self, x, style1, style2):
        for idx, layer in enumerate(self.layers):
            if idx % 2 == 0: style = style1
            else: style = style2
            x = layer(x, style)
        if self.upsample is not None:
            x = self.upsample(x)
        return x
class Generator(nn.Module):
    def __init__(self, style_dim, n_mapping_networks, lr_mlp=0.01, ch_mul=1, size=1024, qk_scale=None, qkv_bias=True, mlp_ratio=4.0, act_type="gelu", drop=0.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        blocks = []
        for _ in range(n_mapping_networks):
            blocks.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"))
        self.mapping_network = nn.Sequential(*blocks)

        start = 2
        end = int(math.log(size, 2))  # 10
        depths = [2] * 9  # 논문 Table 8에 9개의 stageblock에 styleswinblock 2개씩.
        dims = [512, 512, 512, 512, 256 * ch_mul, 128 * ch_mul, 64 * ch_mul, 32 * ch_mul, 16 * ch_mul]
        window_sizes = [4,8,8,8,8,8,8,8,8]  # 논문 table8 참조
        n_heads_lst = [16,16,16,16,8,4,4,4,4]

        self.c_input = ConstantInput(dims[0])  # [BS x 512 x 4 x 4]
        self.layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.n_layers = 0
        for layer_idx in range(start, end+1):
            dim = dims[layer_idx - start]
            window_size = window_sizes[layer_idx - start]
            in_res = 2**layer_idx
            n_heads = n_heads_lst[layer_idx-start]
            depth=depths[layer_idx-start]
            upsample = True if layer_idx < end else None
            out_dim = dims[layer_idx - start + 1] if layer_idx < end else None
            layer = StageBlock(
                dim=dim,
                style_dim=style_dim,
                window_size=window_size,
                in_res=in_res,
                n_heads=n_heads,
                depth=depth,
                qk_scale=qk_scale,
                qkv_bias=qkv_bias,
                mlp_ratio=mlp_ratio,
                act_type=act_type, 
                drop=drop,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                upsample=upsample, 
                out_dim=out_dim          
            )
            self.layers.append(layer)
            out_dim = dims[layer_idx - start + 1] if layer_idx < end else dims[layer_idx - start]
            to_rgb = ToRGB(out_dim, is_upsample=upsample)
            self.to_rgbs.append(to_rgb)
            self.n_layers += 2
    def forward(self, z):
        style = self.mapping_network(z)  # [BS x style_dim] 
        x = self.c_input(style)
        BS, C, H, W = x.shape
        x = x.permute(0,2,3,1).contiguous().reshape(BS, H*W, C)
        
        skip = None
        for idx, (layer, to_rgb) in enumerate(zip(self.layers, self.to_rgbs)):
            x = layer(x, style, style)
            BS, N, C = x.shape
            H = W = int(math.sqrt(N))
            skip = to_rgb(x.transpose(1,2).reshape(BS, C, H, W), skip)
        return skip       

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.register_buffer('kernel', kernel)
        self.pad = pad
    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out
class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None
    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        return out
class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope
    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)
        return out * math.sqrt(2)
class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        sn=False
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2

        if sn:
            # Not use equal conv2d when apply SN
            layers.append(
                spectral_norm(nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                ))
            )
        else:
            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )
        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))
        super().__init__(*layers)
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, sn=False):
        super().__init__()
        self.conv1 = ConvLayer(in_ch, in_ch, 3, sn=sn)
        self.conv2 = ConvLayer(in_ch, out_ch, 3, downsample=True, sn=sn)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
class FromRGB(nn.Module):
    def __init__(self, out_ch, downsample=True, blur_kernel=[1,3,3,1], sn=False):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.iwt = InverseHaarTransform(3)
            self.downsample = Downsample(blur_kernel)
            self.dwt = HaarTransform(3)
        self.conv = ConvLayer(4*3, out_ch, 1, sn=sn)
    def forward(self, x, skip=None):
        if self.downsample:
            x = self.iwt(x)
            x = self.downsample(x)
            x = self.dwt(x)
        out = self.conv(x)
        if skip is not None:
            out = out + skip
        return x, out
def get_haar_wavelet(in_channels):
    haar_wav_l = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h
    
    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh
class HaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        ll, lh, hl, hh = get_haar_wavelet(in_channels)
    
        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)
        
    def forward(self, input):
        ll = upfirdn2d(input, self.ll, down=2)
        lh = upfirdn2d(input, self.lh, down=2)
        hl = upfirdn2d(input, self.hl, down=2)
        hh = upfirdn2d(input, self.hh, down=2)
        
        return torch.cat((ll, lh, hl, hh), 1)
class InverseHaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', -lh)
        self.register_buffer('hl', -hl)
        self.register_buffer('hh', hh)
        
    def forward(self, input):
        ll, lh, hl, hh = input.chunk(4, 1)
        ll = upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0))
        
        return ll + lh + hl + hh
class Discriminator(nn.Module):
    def __init__(self, ch_mul=2, size=1024, sn=False):
        super().__init__()
        channels = {
            1024: 16 * ch_mul, 
            512 : 32 * ch_mul,
            256 : 64 * ch_mul,
            128 : 128 * ch_mul,
            64 : 256 * ch_mul,
            32 : 512,
            16 : 512,
            8 : 512,
            4 : 512
        }
        self.dwt = HaarTransform(3)
        self.from_rgbs = nn.ModuleList()
        self.convs = nn.ModuleList()

        log_size = int(math.log(size, 2)) - 1  # 9
        in_ch = channels[size]

        for i in range(log_size, 2, -1):  # 1024기준 7번 돈다. downsample은 
            out_ch = channels[2**(i-1)]
            downsample = True if i != log_size else False
            self.from_rgbs.append(FromRGB(in_ch, downsample, sn=sn))
            self.convs.append(ConvBlock(in_ch, out_ch, sn=sn))
            in_ch = out_ch
        self.from_rgbs.append(FromRGB(channels[4], sn=sn))

        self.stddev_group = 4
        self.stddev_feat = 1
        self.final_conv = ConvLayer(in_ch+1, channels[4], 3, sn=sn)
        if sn:
            self.final_linear = nn.Sequential(
                spectral_norm(nn.Linear(channels[4]*4*4, channels[4])),
                FusedLeakyReLU(channels[4]),
                spectral_norm(nn.Linear(channels[4], 1))
            )
        else:
            self.final_linear = nn.Sequential(
                EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
                EqualLinear(channels[4], 1)
            )
    def forward(self, x):
        x = self.dwt(x)
        out = None
        for from_rgb, conv in zip(self.from_rgbs, self.convs):
            x, out = from_rgb(x, skip=out)
            out = conv(out)
        # x : [BS x 12 x 8 x 8], out : [BS x 512 x 4 x 4]
        _, out = self.from_rgbs[-1](x, out)  # 
        BS, C, H, W = out.shape
        group = min(BS, self.stddev_group)
        stddev = out.reshape(BS, -1, self.stddev_feat, C//self.stddev_feat, H, W)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2,3,4], keepdim=True).squeeze(2)
        stddev = stddev.repeat(group, 1, H, W)
        out = torch.cat([out, stddev], dim=1)
        out = self.final_conv(out)
        out = out.reshape(BS, -1)
        out = self.final_linear(out)
        return out


