import torch
import torch.nn as nn

from .base_network import define_network_G
from .base_module import BaseModule

class Vid2VidModelG(BaseModule):  # 5차원 처리를 위한 모델. G는 4차원만 처리를 해준다. 
    def __init__(self, args):
        super().__init__(args)
        self.tG =args.n_frames_G  # 한번에 몇 프레임씩 보는지. 많이 사용한다.
        self.G_input_ch = args.input_ch * self.tG  # G는 4차원으로 들어간다. 한번에 보는 프레임 수 x 채널
        self.G_output_ch = args.output_ch  # 생성되는 1장에 대한 채널수
        self.G_prev_output_ch = (self.tG - 1) * args.input_ch  # 이전에 들어간 프레임 수 x 채널
        self.netG = define_network_G(args, input_ch=self.G_input_ch, output_ch=self.G_output_ch, prev_output_ch=self.G_prev_output_ch)

    def forward(self, real_As, real_Bs, gene_Bs_prev):
        '''
        reaL_A : 도메인 A의 실제 이미지 3장 [BS x T x C x H x W]
        real_B : 도메인 B의 실제 이미지 3장 [BS x T x C x H x W]
        gene_B_prev : 생성 B 이미지 2장. 처음 경우는 생성된 것이 없으므로 real_B를 2장 사용. None값이 들어온다.  [BS x T-1 x C x H x W]
        '''
        if gene_Bs_prev is None:  # 처음 forward일떄
            gene_Bs_prev = real_Bs[:, :(self.tG-1)]
        BS, T, C, H, W = real_As.shape
        real_As_reshaped = real_As.reshape(BS, -1, H, W)
        gene_Bs_prev_reshaped = gene_Bs_prev.reshape(BS, -1, H, W).detach()

        gene_B, gene_flow, gene_weight, gene_B_raw = self.netG(real_As_reshaped, gene_Bs_prev_reshaped)
        gene_B = gene_B.unsqueeze(1)
        real_A = real_As[:, -1:] 
        real_Bs_last2 = real_Bs[:, -2:]  # realB마지막 2개
        gene_flow = gene_flow.unsqueeze(1)
        gene_weight = gene_weight.unsqueeze(1)
        gene_B_raw = gene_B_raw.unsqueeze(1)        
        gene_Bs_prev = gene_Bs_prev.detach()

        '''
        return은 모두 5차원
        gene_B :        [BS x 1 x C x H x W]
        gene_B_raw :    [BS x 1 x C x H x W]
        gene_flow :     [BS x 1 x 2 x H x W]
        gene_weight :   [BS x 1 x 1 x H x W]
        real_A :        [BS x 1 x c x H x W]
        real_Bs_last2 : [BS x 2 x C x H x W]
        gene_Bs_prev :  [BS x t-1 x C x H x W]
        '''        
        return gene_B, gene_B_raw, gene_flow, gene_weight, real_A, real_Bs_last2, gene_Bs_prev
    def inference(self, real_As, gene_Bs_prev):
        with torch.no_grad():
            BS, T, C, H, W = real_As.shape
            real_As_reshaped = real_As.reshape(BS, -1, H, W)
            gene_Bs_prev_reshaped = gene_Bs_prev.reshape(BS, -1, H, W).detach()
            gene_B, gene_flow, gene_weight, gene_B_raw = self.netG(real_As_reshaped, gene_Bs_prev_reshaped)

            # 모두 4차원
            return gene_B, gene_B_raw, gene_flow, gene_weight
    def load(self, load_path):
        load_state_dict = torch.load(load_path)
        self.load_state_dict(load_state_dict["G"])
        print(f"model G is successfully loaded from {load_path}")
    def to_eval(self):
        self.eval()