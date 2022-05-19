import torch
import torch.nn as nn


from .base_model import BaseModel, D_reshape
from .base_module import define_module_G
from .base_network import define_network_D
from .base_network import GANLoss, VGGLoss, MaskedL1Loss
from .flownet import FlowNet

class Vid2VidModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.G = define_module_G(args).cuda(args.local_rank)
        D_input_ch = args.input_ch + args.output_ch
        self.netD = define_network_D(args, D_input_ch).cuda(args.local_rank)
        D_input_ch = args.output_ch * args.n_frames_D + 2 * (args.n_frames_D-1)  # temporal은 나중에 스킵된 프레임들 모으는데, 입력이 시간 고려한 5차원 이미지와 flow이미지가 들어간다. 2는 flow의 채널차원.
        for s in range(args.n_scales_temporal):
            netD_T = define_network_D(args, D_input_ch).cuda(args.local_rank)
            setattr(self, f"netD_T_{s}", netD_T)
            optimizer_D_T = torch.optim.Adam(netD_T.parameters(), lr=args.lr, betas=args.betas)
            setattr(self, f"optimizer_D_T_{s}", optimizer_D_T)        
        self.FlowNet = FlowNet(args).cuda(args.local_rank)
        self.criterion_adv = GANLoss(use_lsgan=not args.no_lsgan, target_real_label=args.target_real_label, target_gene_label=args.target_gene_label)
        self.criterion_flow = MaskedL1Loss()
        self.criterion_warp = MaskedL1Loss()
        self.criterion_feat = nn.L1Loss()
        self.criterion_VGG = VGGLoss(args.local_rank)

        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=args.lr, betas=args.betas)
    def set_input(self, real_As, real_Bs, is_first=False):
        assert real_As.shape == real_Bs.shape, f"input shape are not equal {real_As.shape} vs {real_Bs.shape}"
        self.real_As = real_As
        self.real_Bs = real_Bs
        if is_first: 
            self.gene_Bs_prev = None
            self.frames_all = [None, None, None, None]  # real_B_all, gene_B_all, flow_ref_all, conf_ref_all temporal을 위한 변수임. 지금까지 진행한 프레임들 저장.
    def train(self, iter):  
        gene_B, gene_B_raw, gene_flow, gene_weight, real_A, real_Bs_last2, gene_Bs_prev = self.G(self.real_As, self.real_Bs, self.gene_Bs_prev)


        real_B_prev = real_Bs_last2[:, :-1]  # real_Bs에서 마지막에서 2번쨰
        real_B = real_Bs_last2[:, 1:]  # real_Bs에서 마지막
        flow_ref, conf_ref = self.FlowNet(real_B, real_B_prev)
        if self.gene_Bs_prev is None:  # 처음이면 real_Bs에서 뽑음
            gene_B_prev = real_B_prev
        else:
            gene_B_prev = self.gene_Bs_prev[:, -1:]
        self.gene_Bs_prev = gene_Bs_prev

        # 이미지 저장용
        self.real_A_img = real_A[:, 0].detach()
        self.real_B_img = real_B[:, 0].detach()
        self.gene_B_img = gene_B[:, 0].detach()

        #### D ####
        # print(f"real_B : {real_B.shape}")
        # print(f"gene_B : {gene_B.shape}")
        # print(f"gene_B_raw : {gene_B_raw.shape}")
        # print(f"real_A : {real_A.shape}")
        # print(f"real_B_prev : {real_B_prev.shape}")
        # print(f"gene_B_prev : {gene_B_prev.shape}")
        # print(f"gene_flow : {gene_flow.shape}")
        # print(f"gene_weight : {gene_weight.shape}")
        # print(f"flow_ref : {flow_ref.shape}")
        # print(f"conv_ref : {conf_ref.shape}")

        # 5차원 -> 4차원
        real_B, gene_B, gene_B_raw, real_A, real_B_prev, gene_B_prev, gene_flow, gene_weight, flow_ref, conf_ref = D_reshape([real_B, gene_B, gene_B_raw, real_A, real_B_prev, gene_B_prev, gene_flow, gene_weight, flow_ref, conf_ref])

        real_AB = torch.cat([real_A, real_B], dim=1)
        gene_AB = torch.cat([real_A, gene_B], dim=1)
        real_AB_raw = torch.cat([real_A, real_B], dim=1)
        gene_AB_raw = torch.cat([real_A, gene_B_raw], dim=1)
        #### D loss ####
        # 1. real, gene adv loss
        # 2. gene_raw에 대해서 똑같이
        pred_real = self.netD(real_AB)
        pred_gene = self.netD(gene_AB.detach())
        loss_D_real = self.criterion_adv(pred_real, True)
        loss_D_gene = self.criterion_adv(pred_gene, False)
        
        pred_real_raw = self.netD(real_AB_raw)
        pred_gene_raw = self.netD(gene_AB_raw.detach())
        loss_D_real += self.criterion_adv(pred_real_raw, True)
        loss_D_gene += self.criterion_adv(pred_gene_raw, False)        

        #### G loss ####
        # 1. VGG loss
        # 2. adv loss
        # 3. feat loss
        # 4. warp loss 생성된 이미지 gene_B와 이전 생성된 이미지를 정답 flow로 warp해서 만든 이미지와 비교함
        # 5. flow loss flownet을 정답으로 하여 G에서 생성한 gene_flow를 맞춰줌
        # 6. weight loss이거는 official 구현에서는 0이다. 없음.
        # 7. raw이미지에 대해서도 똑같이 해준다.
        loss_G_VGG = self.criterion_VGG(gene_B, real_B) * self.args.lambda_vgg
        pred_gene = self.netD(gene_AB)
        loss_G_adv = self.criterion_adv(pred_gene, True)

        loss_G_feat = 0
        feat_weights = 4.0 / (self.args.n_layers_D+1)
        D_weights = 1.0 / self.args.n_D
        for i in range(min(len(pred_gene), self.args.n_D)):
            for j in range(len(pred_gene[i])-1):
                loss_G_feat += self.args.lambda_feat * D_weights * feat_weights * self.criterion_feat(pred_gene[i][j], pred_real[i][j].detach()) 

        gene_B_warp_ref = self.resample(gene_B_prev, flow_ref)
        loss_G_gene_warp = self.args.lambda_warp * self.criterion_warp(gene_B, gene_B_warp_ref.detach(), conf_ref)

        loss_G_flow = self.args.lambda_flow * self.criterion_flow(gene_flow, flow_ref, conf_ref)
        real_B_warp = self.resample(real_B_prev, gene_flow)
        loss_G_real_warp = self.args.lambda_warp * self.criterion_flow(real_B_warp, real_B, conf_ref)
        
        loss_G_weight = torch.zeros_like(gene_weight)

        # raw이미지 한번 더
        loss_G_VGG += self.criterion_VGG(gene_B_raw, real_B) * self.args.lambda_vgg
        pred_gene_raw = self.netD(gene_AB_raw)
        loss_G_adv += self.criterion_adv(pred_gene_raw, True)
        for i in range(min(len(pred_gene_raw), self.args.n_D)):
            for j in range(len(pred_gene_raw[i])-1):
                loss_G_feat += self.args.lambda_feat * D_weights * feat_weights * self.criterion_feat(pred_gene[i][j], pred_real[i][j].detach())
        
        losses = [loss_D_real, loss_D_gene, loss_G_VGG, loss_G_adv, loss_G_feat, loss_G_gene_warp, loss_G_flow, loss_G_real_warp, loss_G_weight]
        losses = [torch.mean(l) for l in losses]

        #### temporal : 지금까지 진행한 것들을 몇 프레임씩 스킵으로 n_scales_temporal만큼 추가 훈련 ####
        self.frames_all, frames_all_skipped = self.get_all_skipped_frames(self.frames_all, real_B, gene_B, flow_ref, conf_ref)
        
        # frames_all_skipped는 [real, gene, flow, conf]를 담고있고 각 원소는 길이가 n_scales_temporal이하이다. 하지만 모두 길이는 같으므로 아래처럼 원소의 각 위치에서 한개씩 뽑아서 로스 계산.
        loss_D_T_lst_lst = []  # D_T는 개별적으로 들어가야 하기 때문에 개별 리스트로 모아야됨.
        loss_G_T_lst = []  # G_T는 loss_G에 통합되므로 상관없다.
        for s in range(self.args.n_scales_temporal):
            if frames_all_skipped[0][s] is not None:
                D_T_inputs = [fs[s] for fs in frames_all_skipped]
                
                real_B_T, gene_B_T, flow_ref_T, conf_ref_T = D_T_inputs  # 위에서 skipped된 프레임들을 모았으므로 5차원이다. 
                BS, T, C, H, W = real_B_T.shape
                netD_T = getattr(self, f"netD_T_{s}")
                real_B_T = real_B_T.reshape(BS, T*C, H, W)  # [BS x 9 x 512 x 512]
                gene_B_T = gene_B_T.reshape(BS, T*C, H, W)  # [BS x 9 x 512 x 512]
                BS, T, C, H, W = flow_ref_T.shape
                flow_ref_T = flow_ref_T.reshape(BS, T*C, H, W)
                real_B_T = torch.cat([real_B_T, flow_ref_T], dim=1)
                gene_B_T = torch.cat([gene_B_T, flow_ref_T], dim=1)
                pred_real = netD_T(real_B_T)
                pred_gene = netD_T(gene_B_T.detach())
                loss_D_T_real = self.criterion_adv(pred_real, True)
                loss_D_T_gene = self.criterion_adv(pred_gene, False)
                
                pred_gene = netD_T(gene_B_T)
                loss_G_T_adv = self.criterion_adv(pred_gene, True)
                loss_G_T_feat = 0
                for i in range(min(len(pred_gene), self.args.n_D)):
                    for j in range(len(pred_gene[i])-1):
                        loss_G_T_feat += self.args.lambda_feat * D_weights * feat_weights * self.criterion_feat(pred_gene[i][j], pred_real[i][j].detach()) 
                loss_D_T_lst = [loss_D_T_real, loss_D_T_gene]
                loss_D_T_lst_lst.append(loss_D_T_lst)
                loss_G_T_lst = [loss_G_T_adv, loss_G_T_feat]
                
        
        #### loss & optimizer ####
        loss_D_lst = [loss_D_real, loss_D_gene]
        loss_G_lst = [loss_G_adv, loss_G_feat, loss_G_VGG, loss_G_real_warp, loss_G_gene_warp, loss_G_flow, loss_G_weight]
        self.loss_D = 0
        for l in loss_D_lst:
            self.loss_D += torch.mean(l) * 0.5
        self.loss_G = 0
        for l in loss_G_lst:
            self.loss_G += torch.mean(l)
        for l in loss_G_T_lst:
            self.loss_G += torch.mean(l)
        self.loss_D_T_lst = []
        for l_lst in loss_D_T_lst_lst:
            loss_tmp = 0
            for l in l_lst:
                loss_tmp += torch.mean(l) * 0.5
            self.loss_D_T_lst.append(loss_tmp)
        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        for idx, loss in enumerate(self.loss_D_T_lst):
            optimizer_D_T = getattr(self, f"optimizer_D_T_{idx}")
            optimizer_D_T.zero_grad()
            loss.backward()
            optimizer_D_T.step()
    def get_all_skipped_frames(self, frames_all, real_B, gene_B, flow_ref, conf_ref):
        real_B_all, gene_B_all, flow_ref_all, conf_ref_all = frames_all 
        real_B = real_B.unsqueeze(1)
        gene_B = gene_B.unsqueeze(1)
        flow_ref = flow_ref.unsqueeze(1)
        conf_ref = conf_ref.unsqueeze(1)

        real_B_all, real_B_skipped = self.get_skipped_frames(real_B_all, real_B, self.args.n_scales_temporal)
        gene_B_all, gene_B_skipped = self.get_skipped_frames(gene_B_all, gene_B, self.args.n_scales_temporal)
        flow_ref_all, conf_ref_all, flow_ref_skipped, conf_ref_skipped = self.get_skipped_flows(flow_ref_all, conf_ref_all, real_B_skipped, flow_ref, conf_ref)

        frames_all = (real_B_all, gene_B_all, flow_ref_all, conf_ref_all)
        frames_all_skipped = (real_B_skipped, gene_B_skipped, flow_ref_skipped, conf_ref_skipped)
        return frames_all, frames_all_skipped       
    def get_skipped_frames(self, B_all, B, nst):
        '''
        nst : n_scales_temporal, real과 flow가 다르다. flow는 이전만 고려하므로 1이 들어감.
        '''
        n_frames_D = self.args.n_frames_D
        B_all = torch.cat([B_all.detach(), B], dim=1) if B_all is not None else B
        B_skipped = [None] * nst
        for s in range(nst):
            tDs = n_frames_D ** s  # 그냥 넓게 하기위해 한다. 
            span = tDs * (n_frames_D - 1)
            n_groups = min(B_all.shape[1] - span, B.shape[1])
            if n_groups > 0:
                for t in range(0, n_groups, n_frames_D):
                    skip = B_all[:, (-span-t-1):-t:tDs].contiguous() if t != 0 else B_all[:, -span-1::tDs].contiguous()
                    B_skipped[s] = torch.cat([B_skipped[s], skip]) if B_skipped[s] is not None else skip
        max_prev_frames = n_frames_D ** (nst - 1) * (n_frames_D - 1)
        if B_all.shape[1] > max_prev_frames:
            B_all = B_all[:, -max_prev_frames:]
        return B_all, B_skipped
    def get_skipped_flows(self, flow_ref_all, conf_ref_all, real_B, flow_ref, conf_ref):
        nst = self.args.n_scales_temporal
        flow_ref_skipped = [None] * nst
        conf_ref_skipped = [None] * nst
        flow_ref_all, flows = self.get_skipped_frames(flow_ref_all, flow_ref, 1)
        conf_ref_all, confs = self.get_skipped_frames(conf_ref_all, conf_ref, 1)
        if flows[0] is not None:
            flow_ref_skipped[0], conf_ref_skipped[0] = flows[0][:,1:], confs[0][:,1:]

        for s in range(1, nst):
            if real_B[s] is not None and real_B[s].shape[1] == self.args.n_frames_D:
                flow_ref_skipped[s], conf_ref_skipped[s] = self.FlowNet(real_B[s][:, 1:], real_B[s][:, :-1])
        return flow_ref_all, conf_ref_all, flow_ref_skipped, conf_ref_skipped
    def save(self, to_path):
        state_dict = {}
        if self.args.use_DDP:
            state_dict["netG"] = self.G.netG.module.state_dict()
            state_dict["netD"] = self.netD.module.state_dict()
            for s in range(self.args.n_scales_temporal):
                state_dict[f"netD_T_{s}"] = getattr(self, f"netD_T_{s}").module.state_dict()
        else:
            state_dict["netG"] = self.G.netG.state_dict()
            state_dict["netD"] = self.netD.state_dict()
            for s in range(self.args.n_scales_temporal):
                state_dict[f"netD_T_{s}"] = getattr(self, f"netD_T_{s}").state_dict()
        torch.save(state_dict, to_path)