import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

def window_partition(x, window_size):  # [BS x H x W x C] -> [(BS*가로갯수*세로갯수) x window x window x C]
    BS, H, W, C = x.shape
    x = x.reshape(BS, H//window_size, window_size, W//window_size, window_size, C)
    x = x.permute(0,1,3,2,4,5).contiguous().reshape(-1, window_size, window_size, C)
    return x
def window_reverse(x, window_size, H, W):  # [(BS*가로갯수*세로갯수) x window x window x C] -> [BS x H x W x C]
    n_H = H // window_size
    n_W = W // window_size
    BS = x.shape[0] // (n_H*n_W)
    x = x.reshape(BS, n_H, n_W, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().reshape(BS, H, W, -1)
    return x    
class CustomNorm(nn.Module):
    def __init__(self, norm_type, dim):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == "ln": self.norm_layer = nn.LayerNorm(dim)
    def forward(self, x):
        if self.norm_type == "none": return x
        else: return self.norm_layer(x)
class CustomAct(nn.Module):
    def __init__(self, act_type):
        super().__init__()
        if act_type == "gelu": self.act_layer = nn.GELU()
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
class PatchConv(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_ch=3, embed_dim=96, norm_type=None):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_type is not None: self.norm_layer = CustomNorm(norm_type, embed_dim)
        else: self.norm_layer = None
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1,2)
        if self.norm_layer: x = self.norm_layer(x)
        return x
class WindowAttention(nn.Module):  # 이것의 입력은 window_partition함수를 거쳣기 때문에 [(BSx가로갯수x세로갯수)x window_size**2 x C].
    def __init__(self, dim, window_size, n_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.window_size = window_size
        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.qk_scale = qk_scale or head_dim ** -0.5
        self.qkv_layer = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # relative position encoding
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # meshgrid 하면 [win x win] 행렬의 위치 좌표를 튜플로 첫번째는 x의 좌표들, 두번째는 y의 좌표들 나오는데 이를 stack해서 [2 x win x win]
        coords = coords.flatten(1)  # [2 x win**2]
        relative_coords = coords[:, :, None] - coords[:, None, :]  # [2 x win**2 x win**2]
        relative_coords = relative_coords.permute(1,2,0).contiguous()  # [win**2 x win**2 x 2]
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2*window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [win**2 x win**2]
        self.register_buffer("relative_position_index", relative_position_index)

        self.relative_position_bias_table = nn.Parameter(torch.zeros(((2*window_size - 1)**2, n_heads)))  # [(2*win-1) ** 2 x n_heads]
        trunc_normal_(self.relative_position_bias_table, std=0.02)
    def forward(self, x, mask=None):  # mask는 -100과 0으로 이루어진 attn_mask가 들어간다. 
        BS, N, C = x.shape  # BS = BS*가로*세로, N == window**2
        qkv = self.qkv_layer(x).reshape(BS, N, 3, self.n_heads, C//self.n_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q@k.transpose(-2,-1)) * self.qk_scale  # [BS x n_heads x N x N]

        # relative position encoding
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]  # [(win**2)**2 x n_heads]
        relative_position_bias = relative_position_bias.reshape(self.window_size**2, self.window_size**2, self.n_heads)  # [win**2 x win**2 x n_heads]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [n_heads x win**2 x win**2]
        attn = attn + relative_position_bias.unsqueeze(0)  # [BS x n_heads x window**2 x window**2]

        if mask is not None:  # mask : [(가로*세로) x window_size**2 x window_size**2]
            n_windows = mask.shape[0]
            attn = attn.reshape(BS // n_windows, n_windows, self.n_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_heads, N, N)       
        attn = F.softmax(attn, dim=-1)  # [BS x n_heads x N x N]
        attn = self.attn_drop(attn)
        x = attn@v  # [BS x n_heads x N x C//n_heads]
        x = x.transpose(1, 2)  # [BS x N x n_heads x C//n_heads]
        x = x.reshape(BS, N, C)  
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class PatchMerging(nn.Module):  # [BS x H*W x C] => [BS x H//2 x W//2 x 4C]
    def __init__(self, dim, in_res, norm_type="ln"):
        super().__init__()
        self.in_res = in_res
        assert self.in_res % 2 == 0
        self.norm_layer = CustomNorm(norm_type, 4*dim)
        self.reduction_layer = nn.Linear(4*dim, 2*dim, bias=False)
    def forward(self, x):  # x : [BS x H*W x C]
        BS, N, C = x.shape
        assert N == self.in_res ** 2
        x = x.reshape(BS, self.in_res, self.in_res, C)
        x1 = x[:, 0::2, 0::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, :]
        x = torch.cat([x1,x2,x3,x4], dim=-1)  # [BS x H//2 x W//2 x 4C]
        x = x.reshape(BS, -1, 4*C)
        
        x = self.norm_layer(x)
        x = self.reduction_layer(x)
        return x       
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, in_res, window_size, n_heads, norm_type="ln", qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.0, mlp_ratio=4.0, act_type="gelu", drop=0.0, shift_size=0):
        super().__init__()
        self.in_res = in_res
        self.window_size = window_size
        self.shift_size = shift_size  # SW-MSA의 shift 조절. window_size//2로 조절된다.
        self.norm1 = CustomNorm(norm_type, dim)
        self.attn = WindowAttention(dim=dim, window_size=window_size, n_heads=n_heads, qkv_bias=qkv_bias,
        qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = CustomNorm(norm_type, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_dim=dim, hidden_dim=mlp_hidden_dim, act_type=act_type, drop=drop)

        if self.shift_size > 0:
            
            H, W = in_res, in_res
            # 아직 이 마스크가 무슨 의미인지는 잘 모르겠다.. H=W=8, window=4, shift=2로 하면 총 4개가 나오는데 각각 0과 -100의 위치가 다름.
            img_mask = torch.zeros((1,H,W,1))
            h_slices = [
                slice(0, -window_size), 
                slice(-window_size, -shift_size), 
                slice(-shift_size, None)
            ]
            w_slices = [
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None)
            ]
            cnt = 0
            for h_slice in h_slices:
                for w_slice in w_slices:
                    img_mask[:, h_slice, w_slice, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # [(가로*세로) x window_size x window_size x 1]
            mask_windows = mask_windows.reshape(-1, self.window_size**2)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill_(attn_mask!=0, -100.0).masked_fill_(attn_mask==0, 0.0)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)  # [(가로*세로) x window_size**2 x window_size**2]
    def forward(self, x):
        H = W = self.in_res
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # x 를 shift 하는 것은 간단.
        if self.shift_size > 0: 
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
        else: 
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(-1, self.window_size**2, C)

        attn_windows = self.attn(x_windows, self.attn_mask)  # [(BS*가로*세로) x window**2 x C]
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, window_size=self.window_size, H=H, W=W)

        # 다시 역 shift로 맞춰준다.
        if self.shift_size > 0:
            x = torch.roll(shifted_x, (self.shift_size, self.shift_size), dims=(1,2))
        else:
            x = shifted_x

        x = x.reshape(B, N, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class StageBlock(nn.Module):
    def __init__(self, dim, in_res, n_heads, depth, window_size=7, norm_type="ln", qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.0, mlp_ratio=4.0, act_type="gelu", drop=0.0, downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                in_res=in_res, 
                window_size=window_size,
                n_heads=n_heads, 
                norm_type=norm_type,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                mlp_ratio=mlp_ratio,
                act_type=act_type, 
                drop=drop,
                shift_size=0 if (i%2==0) else window_size//2
            ) for i in range(depth)
        ])

        if downsample is not None:
            self.downsample_layer = PatchMerging(dim=dim, in_res=in_res, norm_type=norm_type)
        else:
            self.downsample_layer = None
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample_layer is not None:
            x = self.downsample_layer(x)
        return x      
class SwinTransformerModel(nn.Module):
    def __init__(self, n_cls=1000, img_size=224, patch_size=4, in_ch=3, embed_dim=96, window_size=7, norm_type="ln", patch_norm=True, pos_drop=0.0, drop_path_rate=0.1, depths=[2,2,6,2], n_heads=[3,6,12,24], mlp_ratio=4.0, drop=0.0, qkv_bias=True, qk_scale=None, act_type="gelu", attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.patch_conv = PatchConv(img_size=img_size, patch_size=patch_size, in_ch=in_ch, embed_dim=embed_dim, norm_type=norm_type if patch_norm else "none")
        patch_res = img_size // patch_size
        self.pos_drop = nn.Dropout(pos_drop)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stage_blocks = nn.ModuleList()
        n_stage_blocks = len(depths)
        for idx in range(n_stage_blocks):
            block = StageBlock(
                dim=int(embed_dim*(2**idx)),
                in_res=patch_res//(2**idx),
                n_heads=n_heads[idx],
                depth=depths[idx],
                window_size=window_size,
                norm_type=norm_type,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=dpr[sum(depths[:idx]):sum(depths[:idx+1])],
                downsample=PatchMerging if (idx < n_stage_blocks - 1) else None,
                act_type=act_type,
                drop=drop,
                mlp_ratio=mlp_ratio           
            )
            self.stage_blocks.append(block)
        last_dim =int(embed_dim * (2**(n_stage_blocks-1)))
        self.norm_layer = CustomNorm(norm_type, last_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(last_dim, n_cls)
    def forward_feature(self, x):
        x = self.patch_conv(x)
        x = self.pos_drop(x)

        for block in self.stage_blocks:
            x = block(x)
        
        x = self.norm_layer(x)  # [BS x N x C]
        x = self.avgpool(x.transpose(1,2)).flatten(1)  # [BS x C]
        return x
    def forward(self, x):
        x = self.forward_feature(x)
        x = self.head(x)
        return x



        

