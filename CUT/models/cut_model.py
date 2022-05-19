import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel as DDP

from .base_model import BaseModel
from .networks import define_G, define_D, define_F, GANLoss, PatchNCELoss


class CUTModel(BaseModel):
    def __init__(self, args):
        BaseModel.__init__(self, args)
        self.nce_layers = [int(i) for i in args.nce_layers.split(",")]
        self.G = define_G(args.input_ch, args.output_ch, ngf=args.ngf, netG=args.netG, norm=args.normG, use_dropout=args.use_dropout, init_type=args.init_type).cuda(args.local_rank)
        self.D = define_D(args.input_ch, args.output_ch, netD=args.netD, n_layers_D=args.n_layers_D, norm=args.normD, init_type=args.init_type).cuda(args.local_rank)
        self.F = define_F(args.input_ch, args.netF, norm=args.normG, use_dropout=args.use_dropout, init_type=args.init_type, netF_nc=args.netF_nc).cuda(args.local_rank)
        if args.netF == "mlp_sample":
            with torch.no_grad():
                si = torch.randn((args.batch_size, args.input_ch, args.crop_size, args.crop_size)).cuda(args.local_rank)
                feats = self.G(si, self.nce_layers, encode_only=True)
                self.F.create_mlp(feats)            
        if self.args.use_DDP:
            self.G = DDP(self.G, device_ids=[self.args.local_rank])
            self.F = DDP(self.F, device_ids=[self.args.local_rank])
            self.D = DDP(self.D, device_ids=[self.args.local_rank])
            
        self.criterionGAN = GANLoss(args.gan_mode).cuda(args.local_rank)
        self.criterionNCE = []

        for _ in self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(args).cuda(args.local_rank))
        self.criterionID = nn.L1Loss().cuda(args.local_rank)
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=args.G_lr, betas=args.betas)
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=args.D_lr, betas=args.betas)
    
    
    def set_input(self, real_A, real_B):
        self.real_A = real_A
        self.real_B = real_B
    def train(self):
        bs = self.real_A.shape[0]
        self.real = torch.cat([self.real_A, self.real_B], dim=0)
        self.gene = self.G(self.real)
        self.gene_B = self.gene[:bs]
        self.idt_B = self.gene[bs:]

        # optimize D
        self.set_requires_grad(self.D, requires_grad=True)
        self.set_requires_grad(self.G, requires_grad=False)

        pred_gene_B = self.D(self.gene_B.detach())
        loss_D_gene = self.criterionGAN(pred_gene_B, False).mean()
        pred_real_B = self.D(self.real_B)
        loss_D_real = self.criterionGAN(pred_real_B, True).mean()

        self.loss_D_adv = (loss_D_gene + loss_D_real) * 0.5
        self.optimizer_D.zero_grad()
        self.loss_D_adv.backward()
        self.optimizer_D.step()

        # optimize G
        self.set_requires_grad(self.G, requires_grad=True)
        self.set_requires_grad(self.D, requires_grad=False)

        pred_gene_B = self.D(self.gene_B)
        self.loss_G_adv = self.criterionGAN(pred_gene_B, True).mean() * self.args.lambda_GAN

        self.loss_G_NCE_A = self.calc_NCE_loss(self.real_A, self.gene_B)
        self.loss_G_NCE_B = self.calc_NCE_loss(self.real_B, self.idt_B)
        self.loss_G_NCE = (self.loss_G_NCE_A + self.loss_G_NCE_B) * 0.5

        self.loss_G = self.loss_G_adv + self.loss_G_NCE
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

    def inference(self, real_A):
        with torch.no_grad():
            gene_img = self.G(real_A)
        return gene_img
    def data_dependent_init(self, real_A, real_B):  # 단순히 netF를 init하기 위해서 하는 작업. 백워드는 안한다.
        bs = real_A.shape[0]
        self.set_input(real_A, real_B)
        self.real = torch.cat([self.real_A, self.real_B], dim=0)  # 한방에 forward
        self.gene = self.G(self.real)
        self.gene_B = self.gene[:bs]  
        self.idt_B = self.gene[bs:]

        # compute D loss
        pred_gene_B = self.D(self.gene_B.detach())
        loss_D_gene = self.criterionGAN(pred_gene_B, False).mean()
        pred_real_B = self.D(self.real_B)
        loss_D_real = self.criterionGAN(pred_real_B, True).mean()
        self.loss_D_adv = (loss_D_gene + loss_D_real) * 0.5

        # compute G loss
        pred_gene_B = self.D(self.gene_B)
        self.loss_G_adv = self.criterionGAN(pred_gene_B, True).mean() * self.args.lambda_GAN
        self.loss_G_NCE_A = self.calc_NCE_loss(self.real_A, self.gene_B)
        self.loss_G_NCE_B = self.calc_NCE_loss(self.real_B, self.idt_B)
        self.loss_G_NCE = (self.loss_G_NCE_A + self.loss_G_NCE_B) * 0.5
        self.loss_G = self.loss_G_adv + self.loss_G_NCE
        self.optimizer_F = torch.optim.Adam(self.F.parameters(), lr=self.args.G_lr, betas=self.args.betas)
        if self.args.use_DDP:
            self.G = DDP(self.G, device_ids=[self.args.local_rank])
            self.F = DDP(self.F, device_ids=[self.args.local_rank])
            self.D = DDP(self.D, device_ids=[self.args.local_rank])
    def calc_NCE_loss(self, src, target):  # 이거 이름 잘못 된듯. q가 target 이니까 헷갈린다. 
        n_layers = len(self.nce_layers)
        feat_q = self.G(target, self.nce_layers, encode_only=True)
        feat_k = self.G(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.F(feat_k, self.args.num_patches, None)  # 이 버전에서는 0,4,8,12,16번째 레이어에서 뽑음. 총 5개의 feature map
        feat_q_pool, _ = self.F(feat_q, self.args.num_patches, sample_ids)  # feat k의 패치위치는 랜덤, q는 k에 맞춰서 간다.        

        total_nce_loss = 0.0
        for f_q, f_k, cri, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = cri(f_q, f_k) * self.args.lambda_NCE
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers
    def save(self, to_path):
        state_dict = {}
        if self.args.use_DDP:
            state_dict["G"] = self.G.module.state_dict()
            state_dict["D"] = self.D.module.state_dict()
            state_dict["F"] = self.F.module.state_dict()
        else:
            state_dict["G"] = self.G.state_dict()
            state_dict["D"] = self.D.state_dict()
            state_dict["F"] = self.F.state_dict()
        state_dict["optimizer_G"] = self.optimizer_G.state_dict()
        state_dict["optimizer_D"] = self.optimizer_D.state_dict()
        state_dict["optimizer_F"] = self.optimizer_F.state_dict()
        if self.args.local_rank == 0:
            torch.save(state_dict, to_path)
    def to_train(self):
        self.G.train()
        self.D.train()
        self.F.train()
    def to_eval(self):
        self.G.eval()
        self.D.eval()
        self.F.eval()