import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helper import CustomNorm, window_partition, window_reverse, trunc_normal_, MLP, SinusoidalPositionalEmbedding

class WindowAttention(nn.Module):
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
        coords = coords.flatten(1)  # [2 x win**2]
        relative_coords = coords[:, :, None] - coords[:, None, :]  # [2 x win**2 x win**2]
        relative_coords = relative_coords.permute(1,2,0).contiguous()  # [win**2 x win**2 x 2]
        relative_coords[:,:,0] += window_size - 1
        relative_coords[:,:,1] += window_size - 1
        relative_coords[:,:,0] += 2*window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(((2*window_size-1)**2, n_heads)))
        trunc_normal_(self.relative_position_bias_table, std=0.02)
    def forward(self, qkv, mask=None):  # qkv : [3 x BS x H x W x C//2] => [BS*nW x win**2 x C//2]forward코드테서는 C//2를 C로 사용할 것이지만, 큰 흐름에서는 double attnetion을 사용하므로 C//2가 맞다. mask : [nW x win**2 x win**2]
        q, k, v = qkv[0], qkv[1], qkv[2]
        assert q.shape == k.shape == v.shape
        BS, H, W, C = q.shape
        q = window_partition(q, self.window_size).reshape(-1, self.window_size**2, C)
        k = window_partition(k, self.window_size).reshape(-1, self.window_size**2, C)
        v = window_partition(v, self.window_size).reshape(-1, self.window_size**2, C)
        BS, N, C = q.shape
        assert N == self.window_size**2
        q = q.reshape(BS, N, self.n_heads, C//self.n_heads).permute(0,2,1,3).contiguous()
        k = k.reshape(BS, N, self.n_heads, C//self.n_heads).permute(0,2,1,3).contiguous()
        v = v.reshape(BS, N, self.n_heads, C//self.n_heads).permute(0,2,1,3).contiguous()

        attn = q@k.transpose(-2,-1) * self.qk_scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]  # [(win**2)**2 x n_heads]
        relative_position_bias = relative_position_bias.reshape(self.window_size**2, self.window_size**2, self.n_heads).permute(2,0,1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)  # [BS*nW x n_heads x win**2 x win**2]

        if mask is not None:  # mask : [nW x win**2 x win**2]
            n_windows = mask.shape[0]
            attn = attn.reshape(BS // n_windows, n_windows, self.n_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(BS, self.n_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = attn@v  # [BS x n_heads x win**2 x C//n_heads]
        x = x.transpose(1,2).reshape(BS, N, C)
        return x
class Block(nn.Module):
    def __init__(self, dim, in_res, window_size, n_heads, norm_type="pn", qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, mlp_ratio=4.0, act_type="gelu", drop=0.0):
        super().__init__()
        self.in_res = in_res
        self.n_heads = n_heads
        self.window_size=window_size
        self.norm1 = CustomNorm(norm_type, dim=dim)
        self.qkv_layer = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn1 = WindowAttention(dim=dim//2, window_size=window_size, n_heads=n_heads//2, qk_scale=qk_scale, attn_drop=attn_drop)
        self.attn2 = WindowAttention(dim=dim//2, window_size=window_size, n_heads=n_heads//2, qk_scale=qk_scale, attn_drop=attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # shift window를 위한 mask 생성
        H = W = in_res
        img_mask = torch.zeros((1, H, W, 1))
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
        attn_mask2 = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [1 x win**2 x win**2]
        attn_mask2 = attn_mask2.masked_fill(attn_mask2 != 0, -100.0).masked_fill(attn_mask2 == 0, 0.0)
        self.register_buffer("attn_mask1", None)
        self.register_buffer("attn_mask2", attn_mask2)
        
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.norm2 = CustomNorm(norm_type, dim)
        self.mlp = MLP(dim, hidden_dim=mlp_hidden_dim, act_type=act_type, drop=drop)
    def forward(self, x):  # x : [BS x N x C] => [BS x N x C]
        H = W = self.in_res
        BS, N, C = x.shape
        assert N == H*W
        shortcut = x
        x = self.norm1(x)
        qkv = self.qkv_layer(x).reshape(BS, N, 3, C).permute(2,0,1,3).contiguous().reshape(3*BS, H, W, C)
        qkv1 = qkv[:,:,:,:C//2]
        qkv2 = qkv[:,:,:,C//2:]
        qkv2 = torch.roll(qkv2, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
        qkv1 = qkv1.reshape(3, BS, H, W, C//2)
        qkv2 = qkv2.reshape(3, BS, H, W, C//2)
        # Double Attention
        x1 = self.attn1(qkv1, self.attn_mask1)  # [BS*nW x win**2 x C//2]
        x2 = self.attn2(qkv2, self.attn_mask2)  # [BS*nW x win**2 x C//2]
        
        x1 = window_reverse(x1, window_size=self.window_size, H=H, W=W)  # [BS x H x W x C//2]
        x2 = window_reverse(x2, window_size=self.window_size, H=H, W=W)  # [BS x H x W x C//2]

        x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(1,2))
        x = torch.cat([x1, x2], dim=-1).reshape(BS, N, C)

        x = self.proj(x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
class EncoderStageBlock(nn.Module):
    def __init__(self, depth, dim, out_dim, in_res, window_size, n_heads, downsample=False, norm_type="pn", qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, mlp_ratio=4.0, act_type="gelu", drop=0.0):
        super().__init__()
        layers = []
        self.in_res = in_res
        for _ in range(depth):
            layers.append(Block(dim=dim, in_res=in_res, window_size=window_size, n_heads=n_heads, norm_type=norm_type, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, mlp_ratio=mlp_ratio, act_type=act_type, drop=drop))
        self.layers = nn.Sequential(*layers)
        if downsample:
            self.downsample = nn.AvgPool2d(kernel_size=(2,2), stride=2)
        else:
            self.downsample = None
        self.reduction = nn.Linear(dim, out_dim)
    def forward(self, x):
        x = self.layers(x)
        if self.downsample: 
            BS, N, C = x.shape
            H = W = self.in_res
            x = x.permute(0, 2, 1).contiguous().reshape(BS, C, H, W)
            x = self.downsample(x)
            x = x.reshape(BS, C, -1).permute(0, 2, 1).contiguous()
        x = self.reduction(x)
        return x     
class BilinearUpsample(nn.Module):
    def __init__(self, in_res, dim, out_dim):
        super().__init__()
        self.in_res = in_res
        self.upsample_layer = nn.Upsample(scale_factor=2, mode="bicubic")
        self.norm = nn.LayerNorm(dim)
        self.reduction = nn.Linear(dim, out_dim)
        self.sin_pos_embed = SinusoidalPositionalEmbedding(embedding_dim=out_dim//2, padding_idx=0, init_size=out_dim//2)
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, x):  # x : [BS x N x C] => [BS x 4N x out_dim]
        H = W = self.in_res
        BS, N, C = x.shape
        x = x.permute(0,2,1).contiguous().reshape(BS, C, H, W)
        x = self.upsample_layer(x)  # [BS x C x 2H x 2W]
        x = x.reshape(BS, C, 4*N).permute(0,2,1).contiguous()
        x = self.norm(x)
        x = self.reduction(x)  # [BS x 4N x out_dim]
        x = x.permute(0,2,1).contiguous().reshape(BS, -1, 2*H, 2*W)  # [BS x 2H x 2W x out_dim]
        x += self.sin_pos_embed.make_grid2d(2*H, 2*W, BS) * self.alpha
        x = x.reshape(BS, 2*H*2*W, -1)
        return x
class DecoderStageBlock(nn.Module):
    def __init__(self, depth, dim, out_dim, in_res, window_size, n_heads, upsample=False, norm_type="pn", qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, mlp_ratio=4.0, act_type="gelu", drop=0.0):
        super().__init__()  
        layers = []
        for _ in range(depth):
            layers.append(Block(dim=dim, in_res=in_res, window_size=window_size, n_heads=n_heads, norm_type=norm_type, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, mlp_ratio=mlp_ratio, act_type=act_type, drop=drop))    
        self.layers = nn.Sequential(*layers)
        if upsample: 
            self.upsample = BilinearUpsample(in_res, dim, out_dim)
        else:
            self.upsample = None
    def forward(self, x):
        x = self.layers(x)
        if self.upsample:
            x = self.upsample(x)
        return x
class Generator(nn.Module):
    def __init__(self, in_ch, 
    depths=[1,3,3,2], 
    dims = [64,128,128,64],
    window_sizes = [4,8,8,8], 
    n_heads_lst = [2,4,4,2], 
    patch_size=8, img_size=512, norm_type="in", qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, mlp_ratio=4.0, act_type="gelu", drop=0.0):
        super().__init__()
        self.img_size = img_size
        self.dims = dims
        self.patch_size = patch_size
        self.patch_conv = nn.Conv2d(in_ch, dims[0], kernel_size=patch_size, stride=patch_size//2,
        padding=patch_size//4)

        self.encoder_1 = EncoderStageBlock(depth=1, dim=dims[0], out_dim=dims[0], in_res=img_size//4, window_size=window_sizes[0], n_heads=n_heads_lst[0],downsample=False, norm_type=norm_type, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, mlp_ratio=mlp_ratio, act_type=act_type, drop=drop)
        self.encoder_2 = EncoderStageBlock(depth=1, dim=dims[0], out_dim=dims[0], in_res=img_size//4, window_size=window_sizes[0], n_heads=n_heads_lst[0],downsample=False, norm_type=norm_type, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, mlp_ratio=mlp_ratio, act_type=act_type, drop=drop)
        self.encoder_3 = EncoderStageBlock(depth=1, dim=dims[0], out_dim=dims[1], in_res=img_size//4, window_size=window_sizes[0], n_heads=n_heads_lst[0],downsample=True, norm_type=norm_type, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, mlp_ratio=mlp_ratio, act_type=act_type, drop=drop)
        self.encoder_4 = EncoderStageBlock(depth=depths[0], dim=dims[1], out_dim=dims[1], in_res=img_size//8, window_size=window_sizes[0], n_heads=n_heads_lst[0],downsample=False, norm_type=norm_type, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, mlp_ratio=mlp_ratio, act_type=act_type, drop=drop)

        self.decoder_1 = DecoderStageBlock(depth=depths[1], dim=dims[1], out_dim=dims[2], in_res=img_size//8, window_size=window_sizes[1], n_heads=n_heads_lst[1], upsample=True, norm_type=norm_type,
        qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, mlp_ratio=mlp_ratio, act_type=act_type, drop=drop)
        self.decoder_2 = DecoderStageBlock(depth=depths[2], dim=dims[2], out_dim=dims[3], in_res=img_size//4, window_size=window_sizes[2], n_heads=n_heads_lst[2], upsample=True, norm_type=norm_type,
        qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, mlp_ratio=mlp_ratio, act_type=act_type, drop=drop)
        self.decoder_3 = DecoderStageBlock(depth=depths[3], dim=dims[3], out_dim=dims[3], in_res=img_size//2, window_size=window_sizes[3], n_heads=n_heads_lst[3], upsample=True, norm_type=norm_type,
        qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, mlp_ratio=mlp_ratio, act_type=act_type, drop=drop)
        self.final_conv = nn.Conv2d(dims[3], in_ch, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, x, layers=[], encode_only=False):
        if not encode_only:
            BS, C, H, W = x.shape
            assert H % self.patch_size == 0
            # encoding stage
            x = self.patch_conv(x)
            BS, C, H, W = x.shape
            x = x.reshape(BS, C, H*W).permute(0, 2, 1).contiguous()
            x = self.encoder_1(x)
            x = self.encoder_2(x)
            x = self.encoder_3(x)
            x = self.encoder_4(x)
            x = self.decoder_1(x)
            x = self.decoder_2(x)
            x = self.decoder_3(x)
            BS, N, C = x.shape
            x = x.permute(0,2,1).contiguous().reshape(BS, C, self.img_size, self.img_size)
            x = self.final_conv(x)
            x = self.tanh(x)
            return x
        else:
            feats = []
            BS, C, H, W = x.shape
            assert H % self.patch_size == 0
            x = self.patch_conv(x)
            BS, C, H, W = x.shape
            feats.append(x)
            x = x.reshape(BS, C, H*W).permute(0, 2, 1).contiguous()
            x = self.encoder_1(x)
            feats.append(x.transpose(-1,-2).reshape(BS, -1, H, W))
            x = self.encoder_2(x)
            feats.append(x.transpose(-1,-2).reshape(BS, -1, H, W))
            x = self.encoder_3(x)
            feats.append(x.transpose(-1,-2).reshape(BS, -1, H//2, W//2))
            x = self.encoder_4(x)
            feats.append(x.transpose(-1,-2).reshape(BS, -1, H//2, W//2))
            return feats
            

        