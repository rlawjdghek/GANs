import torch
import numpy as np
import os
import sys
sys.path.append("D:\\jupyter\\GANs\\Wasserstein GAN GP")
from models.models import Discriminator


def compute_gp_loss(discriminator, p_real, p_fake, device):
    # p_real, p_fake => [B x 1 x 28 x 28] in mnist
    alpha = torch.rand((p_real.shape[0], 1, 1, 1)).to(device)
    interpol = (alpha*p_real + (1-alpha)*p_fake).requires_grad_(True)
    interpol_logit = discriminator(interpol)
    gene_label = torch.ones(interpol_logit.shape).requires_grad_(False).to(device)
    g = torch.autograd.grad(
        outputs=interpol_logit,
        inputs=interpol,
        grad_outputs = gene_label,
        retain_graph=True,  # 그래프가 여러번 backward되어도 안사라짐
        create_graph=True,  # 그래프를 생성해야 이것도 로스로서 유효하게 역전파에 영향을 줄 수 있다.
        only_inputs=True 
    )[0]
    g = g.reshape(g.shape[0], -1)  # 벡터화
    gp = torch.mean((torch.norm(g, p=2, dim=-1) -1)**2)
    return gp
    
    
    
if __name__ == "__main__":
    real_sample = torch.randn((4,1,28,28))
    gene_sample = torch.randn(real_sample.shape)
    discriminator = Discriminator((1,28,28))
    gp_loss = compute_gp_loss(discriminator, real_sample, gene_sample)
    print(f"gp loss : {gp_loss}")