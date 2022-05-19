import math

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_helper import DropPath, trunc_normal_
from .diff_aug import DiffAugment
from torch_utils.ops import upfirdn2d

class PixelNorm(nn.Module):  # [BS x A x embed_dim]가 들어오면 embed_dim의 원소들의 제곱평균을 루트씌워서 나누어줌. 평균을 0으로 하는듯
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=2, keepdim=True) + 1e-8)
class CustomNorm(nn.Module):
    def __init__(self, norm_type, dim):
        super().__init__()
        self.norm_type = norm_type
        self.dim = dim
        if norm_type == "ln": self.norm_layer = nn.LayerNorm(dim)
        elif norm_type == "in": self.norm_layer = nn.InstanceNorm1d(dim)
        elif norm_type == "bn": self.norm_layer = nn.BatchNorm1d(dim)
        elif norm_type == "pn": self.norm_layer = PixelNorm()
    def forward(self, x):
        if self.norm_type == "bn" or self.norm_type == "in":
            x = self.norm_layer(x.permute(0,2,1)).permute(0,2,1)
        else:
            x = self.norm_layer(x)
        return x
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
def windown_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape(B, H//window_size, window_size, W//window_size, window_size, C)
    windows = x.permute(0,1,3,2,4,5).contiguous()  # [BS x 가로갯수 x 세로갯수 x window_size x window_size x C]
    windows = windows.reshape(-1, window_size, window_size, C)  # [BS*가로갯수*세로갯수 x window_size x window_size x C]
    return windows
def window_reverse(windows, window_size, H, W):
    B = windows.shape[0] // (H//window_size * W//window_size)
    x = windows.reshape(B, H//window_size, W//window_size, window_size, window_size, -1)  # 위의 window_partition에서 역순으로 간다.
    x = x.permute(0,1,3,2,4,5).contiguous()
    x = x.reshape(B, H, W, -1)
    return x
def bicubic_upsample(x, H, W):
    BS, N, C = x.shape
    assert H*W == N
    x = x.permute(0,2,1)
    x = x.reshape(BS,C,H,W)
    x = F.interpolate(x, scale_factor=2, mode="bicubic")  # [BS x C x 2H x 2W]
    BS, C, H, W = x.shape
    x = x.reshape(BS, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W
def pixel_upsample(x, H, W):
    BS, N, C = x.shape
    assert H*W == N
    x = x.permute(0,2,1)
    x = x.reshape(BS,C,H,W)
    x = F.pixel_shuffle(x, upscale_factor=2)  # [BS x C//4 x 2H x 2W]  여기서 채널이 줄어든다.
    BS, C, H, W = x.shape
    x = x.reshape(BS, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W
def updown(x, H, W):
    BS, N, C = x.shape
    assert H*W == N
    x = x.permute(0,2,1)
    x = x.reshape(BS,C,H,W)
    x = F.interpolate(x, scale_factor=4, mode="bicubic")  # [BS x C x 4H x 4W]
    x = F.avg_pool2d(x, 4)  # [BS x C x H x W]
    BS, C, H, W = x.shape
    x = x.reshape(BS, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W
class Attention(nn.Module):  # relative position encoding으로 인하여 무조건 x의 두번째 차원은 win**2여야 한다. 
    def __init__(self, dim, n_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, window_size=16):
        super().__init__()
        self.n_heads = n_heads
        self.window_size = window_size
        head_dim = dim // n_heads
        self.qk_scale = qk_scale or head_dim ** -0.5
        self.noise_strength = nn.Parameter(torch.zeros(()))
        self.qkv_layer = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 논문의 Relative Position Encoding
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
    def forward(self, x):
        BS, N, C = x.shape
        x = x + torch.randn((BS, N, 1), device=x.device) * self.noise_strength
        q, k, v = self.qkv_layer(x).reshape(BS, N, 3, self.n_heads, C//self.n_heads).permute(2, 0, 3, 1, 4)
        attn = q@k.transpose(-2, -1) * self.qk_scale  # [BS x n_heads x N x N]

        # relative position encoding
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]  # [(win**2)**2 x n_heads]
        relative_position_bias = relative_position_bias.reshape(self.window_size**2, self.window_size**2, self.n_heads)  # [win**2 x win**2 x n_heads]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [n_heads x win**2 x win**2]
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)  # [BS x n_heads x N x N]
        attn = self.attn_drop(attn)
        x = attn@v  # [BS x n_heads x N x C//n_heads]
        x = x.transpose(1, 2)  # [BS x N x n_heads x C//n_heads]
        x = x.reshape(BS, N, C)  
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class CrossAttention(nn.Module):
    def __init__(self, q_dim, k_dim, n_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.n_heads = n_heads
        head_dim = q_dim//n_heads
        self.qk_scale = qk_scale or head_dim ** -0.5
        self.q_layer = nn.Linear(q_dim, q_dim, bias=qkv_bias)
        self.k_layer = nn.Linear(k_dim, q_dim, bias=qkv_bias)
        self.v_layer = nn.Linear(k_dim, q_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(q_dim, q_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, inputs):
        x, embedding = inputs
        BS, N, C = x.shape
        _, e_N, e_C = embedding.shape
        q = self.q_layer(x)  # [BS x N x q_dim] 사실상 q_dim=k_dim=1024이고 N은 엄샘플 되면서 늘어나는 해상도의 제곱
        k = self.k_layer(embedding)  # [BS x e_N x k_dim] e_N은 celebA에서 8*8이다. k_dim은 q_dim이랑 같은 1024
        v = self.v_layer(embedding)

        q = q.reshape(BS, N, self.n_heads, self.q_dim//self.n_heads).permute(0,2,1,3)  # [BS x n_heads x N x q_dim//n_heads]
        k = k.reshape(BS, e_N, self.n_heads, self.q_dim//self.n_heads).permute(0,2,1,3)  # [BS x n_heads x e_N x k_dim//n_heads]
        v = v.reshape(BS, e_N, self.n_heads, self.q_dim//self.n_heads).permute(0,2,1,3)  # [BS x n_heads x e_N x k_dim//n_heads]

        attn = q@k.transpose(-2, -1) * self.qk_scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        output = attn@v
        output = output.transpose(1,2)
        output = output.reshape(BS, N, self.q_dim)
        output = self.proj(output)
        output = self.proj_drop(output)
        return x + output
class Block(nn.Module):
    def __init__(self, dim, embedding_dim, norm_type="pn", n_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, window_size=16, drop_path=0.0, mlp_ratio=4.0, act_type="gelu", drop=0.0):
        super().__init__()
        self.window_size = window_size
        self.norm_1 = CustomNorm(norm_type, dim)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_2 = CustomNorm(norm_type, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, act_type=act_type, drop=drop)
        self.cross_attention = CrossAttention(q_dim=dim, k_dim=embedding_dim, n_heads=n_heads)
    def forward(self, inputs):
        x, embeddings = inputs
        x = self.cross_attention([x, embeddings])
        BS, N, C = x.shape
        H = W = int(np.sqrt(N))  # Attention layer에는 16밖에 안들어감
        x = x.reshape(BS, H, W, C)
        x = windown_partition(x, self.window_size)  # [BS*가로갯수*세로갯수 x window_size x window_size x C]
        x = x.reshape(-1, self.window_size**2, C)
        x = x + self.drop_path(self.attn(self.norm_1(x)))
        x = x.reshape(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, window_size=self.window_size, H=H, W=W)  # [BS x H x W x C]
        x = x.reshape(BS, N, C)
        x = x + self.drop_path(self.mlp(self.norm_2(x)))
        return [x, embeddings]
class StageBlock(nn.Module):
    def __init__(self, n_blocks, dim, embedding_dim, n_heads, norm_type="pn", qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, window_size=16, drop_path=0.0, mlp_ratio=4.0, drop=0.0, act_type="gelu"):
        super().__init__()
        block = [Block(dim=dim, embedding_dim=embedding_dim, norm_type=norm_type, n_heads=n_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, window_size=window_size, drop_path=drop_path, mlp_ratio=mlp_ratio, drop=drop, act_type=act_type) for _ in range(n_blocks)]
        self.block = nn.Sequential(*block)
    def forward(self, inputs):
        x = self.block(inputs)
        return x
class Generator(nn.Module):
    def __init__(self, latent_dim, bottom_width, embedding_dim, depths, out_ch, n_heads=4, norm_type="pn", qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.0, mlp_ratio=4.0, window_size=16, drop=0.0, act_type="gelu"):
        super().__init__()
        self.bottom_width = bottom_width
        self.embedding_dim = embedding_dim

        self.linear_1 = nn.Linear(latent_dim, bottom_width**2 * embedding_dim)
        self.embedding_layer = nn.Linear(latent_dim, bottom_width**2 * embedding_dim)
        self.embedding_pos = nn.Parameter(torch.zeros(1, bottom_width**2, embedding_dim))  # embedding layer를 거친 텐서와 합해짐.
    
        # 실제 x와 합해짐. upsample은 5번되므로 그것을 기준으로는 6개
        self.pos_embed_1 = nn.Parameter(torch.zeros((1, bottom_width**2, embedding_dim)))
        self.pos_embed_2 = nn.Parameter(torch.zeros((1, (bottom_width*2)**2, embedding_dim)))
        self.pos_embed_3 = nn.Parameter(torch.zeros((1, (bottom_width*4)**2, embedding_dim)))
        self.pos_embed_4 = nn.Parameter(torch.zeros((1, (bottom_width*8)**2, embedding_dim//4)))
        self.pos_embed_5 = nn.Parameter(torch.zeros((1, (bottom_width*16)**2, embedding_dim//16)))
        self.pos_embed_6 = nn.Parameter(torch.zeros((1, (bottom_width*32)**2, embedding_dim//64)))
        # self.pos_embed = [self.pos_embed_1, self.pos_embed_2, self.pos_embed_3, self.pos_embed_4, self.pos_embed_5, self.pos_embed_6]
        trunc_normal_(self.pos_embed_1, std=0.02)
        trunc_normal_(self.pos_embed_2, std=0.02)
        trunc_normal_(self.pos_embed_3, std=0.02)
        trunc_normal_(self.pos_embed_4, std=0.02)
        trunc_normal_(self.pos_embed_5, std=0.02)
        trunc_normal_(self.pos_embed_6, std=0.02)
        self.block_1 = None  # 논문에는 있는데 구현에는 사용하지 않음.
        self.block_2 = StageBlock(
            n_blocks=depths[1], 
            dim=embedding_dim, 
            embedding_dim=embedding_dim, 
            n_heads=n_heads, 
            norm_type=norm_type, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop, 
            proj_drop=proj_drop, 
            window_size=16, 
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            drop=drop,
            act_type=act_type
            )
        self.block_3 = StageBlock(
            n_blocks=depths[2], 
            dim=embedding_dim, 
            embedding_dim=embedding_dim, 
            n_heads=n_heads, 
            norm_type=norm_type, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop, 
            proj_drop=proj_drop, 
            window_size=32, 
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            drop=drop,
            act_type=act_type
            )
        self.block_4 = StageBlock(
            n_blocks=depths[3], 
            dim=embedding_dim//4, 
            embedding_dim=embedding_dim, 
            n_heads=n_heads, 
            norm_type=norm_type, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop, 
            proj_drop=proj_drop, 
            window_size=window_size, 
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            drop=drop,
            act_type=act_type
            )
        self.block_5 = StageBlock(
            n_blocks=depths[4], 
            dim=embedding_dim//16, 
            embedding_dim=embedding_dim, 
            n_heads=n_heads, 
            norm_type=norm_type, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop, 
            proj_drop=proj_drop, 
            window_size=window_size, 
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            drop=drop,
            act_type=act_type
            )
        self.block_6 = StageBlock(
            n_blocks=depths[5], 
            dim=embedding_dim//64, 
            embedding_dim=embedding_dim, 
            n_heads=n_heads, 
            norm_type=norm_type, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop, 
            proj_drop=proj_drop, 
            window_size=window_size, 
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            drop=drop,
            act_type=act_type
            )
        # for pos in self.pos_embed:
        #     trunc_normal_(pos, std=0.02)
        self.last_conv = nn.Conv2d(embedding_dim//64, out_ch, 1, 1, 0)
    def forward(self, z):
        x = self.linear_1(z).reshape(-1, self.bottom_width**2, self.embedding_dim)
        x = x + self.pos_embed_1  # [BS x 8*8 x embedding_dim]
        embedding = self.embedding_layer(z).reshape(-1, self.bottom_width**2, self.embedding_dim)
        embedding = embedding + self.embedding_pos
        
        H, W = self.bottom_width, self.bottom_width
        # block2
        x, H, W = bicubic_upsample(x, H, W)  # [BS x 16*16 x embedding_dim]
        x = x + self.pos_embed_2
        x, _ = self.block_2([x, embedding])
        x, H, W = bicubic_upsample(x, H, W)  # [BS x 32*32 x embedding_dim]
        x = x + self.pos_embed_3
        x, _ = self.block_3([x, embedding])
        x, H, W = pixel_upsample(x, H, W)  # [BS x 64*64 x embedding_dim//4]
        x = x + self.pos_embed_4
        x, _ = self.block_4([x, embedding])
        x, H, W = updown(x, H, W)
        x, H, W = pixel_upsample(x, H, W)  # [BS x 128*128 x embedding_dim//16]
        x = x + self.pos_embed_5
        x, _ = self.block_5([x, embedding])
        x, H, W = updown(x, H, W)
        x, H, W = pixel_upsample(x, H, W)  # [BS x 256*256 x embedding_dim//64]
        x = x + self.pos_embed_6
        x, _ = self.block_6([x, embedding])

        BS, N, C = x.shape
        x = x.permute(0,2,1).reshape(BS, C, H, W)  # [BS x embedding_dim//64 x 256 x 256]
        x = self.last_conv(x)
        return x


wavelets = {
    'haar': [0.7071067811865476, 0.7071067811865476],
    'db1':  [0.7071067811865476, 0.7071067811865476],
    'db2':  [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'db3':  [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'db4':  [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523],
    'db5':  [0.003335725285001549, -0.012580751999015526, -0.006241490213011705, 0.07757149384006515, -0.03224486958502952, -0.24229488706619015, 0.13842814590110342, 0.7243085284385744, 0.6038292697974729, 0.160102397974125],
    'db6':  [-0.00107730108499558, 0.004777257511010651, 0.0005538422009938016, -0.031582039318031156, 0.02752286553001629, 0.09750160558707936, -0.12976686756709563, -0.22626469396516913, 0.3152503517092432, 0.7511339080215775, 0.4946238903983854, 0.11154074335008017],
    'db7':  [0.0003537138000010399, -0.0018016407039998328, 0.00042957797300470274, 0.012550998556013784, -0.01657454163101562, -0.03802993693503463, 0.0806126091510659, 0.07130921926705004, -0.22403618499416572, -0.14390600392910627, 0.4697822874053586, 0.7291320908465551, 0.39653931948230575, 0.07785205408506236],
    'db8':  [-0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771, -0.00487035299301066, 0.008746094047015655, 0.013981027917015516, -0.04408825393106472, -0.01736930100202211, 0.128747426620186, 0.00047248457399797254, -0.2840155429624281, -0.015829105256023893, 0.5853546836548691, 0.6756307362980128, 0.3128715909144659, 0.05441584224308161],
    'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'sym3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'sym4': [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
    'sym5': [0.027333068345077982, 0.029519490925774643, -0.039134249302383094, 0.1993975339773936, 0.7234076904024206, 0.6339789634582119, 0.01660210576452232, -0.17532808990845047, -0.021101834024758855, 0.019538882735286728],
    'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
    'sym7': [0.002681814568257878, -0.0010473848886829163, -0.01263630340325193, 0.03051551316596357, 0.0678926935013727, -0.049552834937127255, 0.017441255086855827, 0.5361019170917628, 0.767764317003164, 0.2886296317515146, -0.14004724044296152, -0.10780823770381774, 0.004010244871533663, 0.010268176708511255],
    'sym8': [-0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298, 0.007607487324917605, -0.1432942383508097, -0.061273359067658524, 0.4813596512583722, 0.7771857517005235, 0.3644418948353314, -0.05194583810770904, -0.027219029917056003, 0.049137179673607506, 0.003808752013890615, -0.01495225833704823, -0.0003029205147213668, 0.0018899503327594609],
}
class Attention2(nn.Module):
    def __init__(self, dim, n_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.n_heads = n_heads
        head_dim = dim//n_heads
        self.qk_scale = qk_scale or head_dim ** -0.5
        self.qkv_layer = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        BS, N, C = x.shape
        q,k,v = self.qkv_layer(x).reshape(BS, N, 3, self.n_heads, C//self.n_heads).permute(2, 0, 3, 1, 4)  
        # [BS x n_heads x N x C//n_heads]
        attn = (q@k.transpose(-2, -1)) * self.qk_scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)  # [BS x n_heads x N x N]

        x = (attn@v).transpose(1, 2)  # [BS x N x n_heads x C//n_heads]
        x = x.reshape(BS, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block2(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, proj_drop=0.0, drop_path=0.0, act_type="gelu", norm_type="ln"):
        super().__init__()
        self.norm_1 = CustomNorm(norm_type, dim)
        self.attn = Attention2(dim, n_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_2 = CustomNorm(norm_type, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_dim=dim, hidden_dim=mlp_hidden_dim, act_type=act_type, drop=drop)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm_1(x)))
        x = x + self.drop_path(self.mlp(self.norm_2(x)))
        return x
class DisBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.0, act_type="gelu", norm_type="ln", drop=0.0):
        super().__init__()
        self.norm_1 = CustomNorm(norm_type, dim)
        self.attn = Attention2(dim, n_heads=n_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_2 = CustomNorm(norm_type, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim=mlp_hidden_dim, act_type=act_type, drop=drop)
        self.gain = np.sqrt(0.5) if norm_type == "none" else 1  
    def forward(self, x):
        x = self.gain * (x + self.drop_path(self.attn(self.norm_1(x))))
        x = self.gain * (x + self.drop_path(self.mlp(self.norm_2(x))))
        return x
class Discriminator(nn.Module):
    def __init__(self, in_ch, diff_aug, n_heads=4, n_cls=1, img_size=256, patch_size=2, embedding_dim=384, drop_rate=0.0, window_size=4, depth=3, qkv_bias=False, qk_scale=None, mlp_ratio=4, drop=0.0, attn_drop=0.0, proj_drop=0.0, act_type="gelu", norm_type="ln", drop_path=0.0):
        super().__init__()
        self.diff_aug = diff_aug
        self.patch_size = patch_size
        self.window_size = window_size

        self.patch_conv_1 = nn.Conv2d(in_ch, embedding_dim//4, kernel_size=patch_size*2, stride=patch_size, padding=patch_size//2)
        self.patch_conv_2 = nn.Conv2d(in_ch, embedding_dim//4, kernel_size=patch_size*2, stride=patch_size*2, padding=0)
        self.patch_conv_3 = nn.Conv2d(in_ch, embedding_dim//2, kernel_size=patch_size*4, stride=patch_size*4, padding=0)
        n_patches_1 = (img_size // patch_size) ** 2
        n_patches_2 = (img_size // (patch_size*2)) ** 2
        n_patches_3 = (img_size // (patch_size*4)) ** 2

        self.cls_token = nn.Parameter(torch.zeros((1,1,embedding_dim)))
        self.pos_embed_1 = nn.Parameter(torch.zeros((1, n_patches_1, embedding_dim//4)))
        self.pos_embed_2 = nn.Parameter(torch.zeros((1, n_patches_2, embedding_dim//2)))
        self.pos_embed_3 = nn.Parameter(torch.zeros((1, n_patches_3, embedding_dim)))

        self.pos_drop = nn.Dropout(drop_rate)
        block_1 = [DisBlock(
            dim=embedding_dim//4, 
            n_heads=n_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop=drop, 
            attn_drop=attn_drop, 
            proj_drop=proj_drop, 
            drop_path=drop_path, 
            act_type=act_type, 
            norm_type=norm_type
            ) for _ in range(depth)]
        block_2 = [DisBlock(
            dim=embedding_dim//2, 
            n_heads=n_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop=drop, 
            attn_drop=attn_drop, 
            proj_drop=proj_drop, 
            drop_path=drop_path, 
            act_type=act_type, 
            norm_type=norm_type
            ) for _ in range(depth-1)]
        block_21 = [DisBlock(
            dim=embedding_dim//2, 
            n_heads=n_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop=drop, 
            attn_drop=attn_drop, 
            proj_drop=proj_drop, 
            drop_path=drop_path, 
            act_type=act_type, 
            norm_type=norm_type
            ) for _ in range(1)]
        block_3 = [DisBlock(
            dim=embedding_dim, 
            n_heads=n_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop=drop, 
            attn_drop=attn_drop, 
            proj_drop=proj_drop, 
            drop_path=drop_path, 
            act_type=act_type, 
            norm_type=norm_type
            ) for _ in range(depth+1)]
        self.block_1 = nn.Sequential(*block_1)
        self.block_2 = nn.Sequential(*block_2)
        self.block_21 = nn.Sequential(*block_21)
        self.block_3 = nn.Sequential(*block_3)
        self.last_block = Block2(embedding_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path, act_type=act_type, norm_type=norm_type)

        self.norm = CustomNorm(norm_type, embedding_dim)
        self.head = nn.Linear(embedding_dim, n_cls)

        trunc_normal_(self.pos_embed_1, std=0.02)
        trunc_normal_(self.pos_embed_2, std=0.02)
        trunc_normal_(self.pos_embed_3, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        if 'filter' in self.diff_aug:
            Hz_lo = np.asarray(wavelets['sym2'])            # H(z)
            Hz_hi = Hz_lo * ((-1) ** np.arange(Hz_lo.size)) # H(-z)
            Hz_lo2 = np.convolve(Hz_lo, Hz_lo[::-1]) / 2    # H(z) * H(z^-1) / 2
            Hz_hi2 = np.convolve(Hz_hi, Hz_hi[::-1]) / 2    # H(-z) * H(-z^-1) / 2
            Hz_fbank = np.eye(4, 1)                         # Bandpass(H(z), b_i)
            for i in range(1, Hz_fbank.shape[0]):
                Hz_fbank = np.dstack([Hz_fbank, np.zeros_like(Hz_fbank)]).reshape(Hz_fbank.shape[0], -1)[:, :-1]
                Hz_fbank = scipy.signal.convolve(Hz_fbank, [Hz_lo2])
                Hz_fbank[i, (Hz_fbank.shape[1] - Hz_hi2.size) // 2 : (Hz_fbank.shape[1] + Hz_hi2.size) // 2] += Hz_hi2
            Hz_fbank = torch.as_tensor(Hz_fbank, dtype=torch.float32)
            self.register_buffer('Hz_fbank', torch.as_tensor(Hz_fbank, dtype=torch.float32))
        else:
            self.Hz_fbank = None
        if 'geo' in self.diff_aug:
            self.register_buffer('Hz_geom', upfirdn2d.setup_filter(wavelets['sym6']))
        else:
            self.Hz_geom = None
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        x = DiffAugment(x, self.diff_aug, True, [self.Hz_geom, self.Hz_fbank])  # [BS x 3 x 256 x 256]
        x_1 = self.patch_conv_1(x).flatten(2).permute(0,2,1)  # [BS x 128*128 x embedding_dim//4]
        x_2 = self.patch_conv_2(x).flatten(2).permute(0,2,1)  # [BS x 64*64 x embedding_dim//2]
        x_3 = self.patch_conv_3(x).flatten(2).permute(0,2,1)  # [BS x 32*32 x embedding_dim]

        BS, _, H, W = x.shape
        H = W = H//self.patch_size
        x = x_1 + self.pos_embed_1  # [BS x 128*128 x embedding_dim//4]
        x = x.reshape(BS, H, W, -1)
        BS, H, W, C = x.shape
        x = windown_partition(x, window_size=self.window_size)  # [BS*가로*세로 x win x win x C]
        x = x.reshape(-1, self.window_size**2, C)  # [BS*가로*세로 x win**2 x C]
        x = self.block_1(x)  # [BS*가로*세로 x win**2 x C]
        x = x.reshape(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W)  # [BS x H x W x C]
        x = x.reshape(BS, H*W, C)
        x = x.permute(0, 2, 1).reshape(BS, C, H, W)

        x = F.avg_pool2d(x, 2)  # [BS x C x H//2 x W//2]
        BS, C, H, W = x.shape
        x = x.flatten(2).permute(0,2,1)  # [BS x 64*64 x embedding_dim//2]
        x = torch.cat([x, x_2], dim=-1)
        x = x + self.pos_embed_2
        x = x.reshape(BS, H, W, -1)
        BS, H, W, C = x.shape
        x = windown_partition(x, self.window_size)
        x = x.reshape(-1, self.window_size**2, C)
        x = self.block_2(x)
        x = x.reshape(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W)  # [BS x 64 x 64 x embedding_dim//2]
        x = x.reshape(BS, H*W, C)
        x = self.block_21(x)  # [BS x 64*64 x embedding_dim//2]
        x = x.permute(0,2,1).reshape(BS, C, H, W)

        x = F.avg_pool2d(x, 2)  # [BS x embedding_dim//2 x 32 x 32]
        BS, C, H, W = x.shape
        x = x.flatten(2).permute(0,2,1)  # [BS x 32*32 x embedding_dim//2]
        x = torch.cat([x, x_3], dim=-1)  # [BS x 32*32 x embedding_dim]
        x = x + self.pos_embed_3
        x = self.block_3(x)  # [BS x 32*32 x embedding_dim]
        
        cls_tokens = self.cls_token.expand(BS, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.last_block(x)
        x = self.norm(x)[:, 0]
        x = self.head(x)
        return x

if __name__ == "__main__":
    class args:
        diff_aug = "filter,translation,erase_ratio,color,hue"
        img_size = 256
    import torch
    import random
    import torch.backends.cudnn as cudnn
