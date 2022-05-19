import torch
import torch.nn as nn
from timm.scheduler.cosine_lr import CosineLRScheduler

from .base_model import BaseModel
from .networks import SwinTransformerModel

class SwinTransformer(BaseModel):
    def __init__(self, args, logger, iters_per_epoch):
        BaseModel.__init__(self, args, logger)
        self.ST_model = SwinTransformerModel()
        self.ST_model.apply(self._weight_init)
        self.ST_model.cuda(args.local_rank)
        if args.use_DDP: 
            self.ST_model = nn.parallel.DistributedDataParallel(self.ST_model, device_ids=[args.local_rank])
        
        self.criterion_CE = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.ST_model.parameters(), eps=args.eps, betas=args.betas, lr=args.lr, weight_decay=args.weight_decay)
        total_iters = int(iters_per_epoch * args.n_epochs)
        warmup_iters = int(iters_per_epoch * args.warmup_epoch)
        self.lr_scheduler = CosineLRScheduler(self.optimizer, t_initial=total_iters, lr_min=args.min_lr, warmup_lr_init=args.warmup_lr, warmup_t=warmup_iters, cycle_limit=1, t_in_epochs=False)
        

    def _weight_init(self, m):
        cls_name = m.__class__.__name__
        if cls_name.find("Conv2d") != -1:
            if self.args.layer_init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif self.args.layer_init_type == "orth":
                nn.init.orthogonal_(m.weight.data)
            elif self.args.layer_init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight.data, 1.0)
            else: raise NotImplementedError(f"initialize {self.args.layer_init_type} is not implemented!!!!")
        elif cls_name.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
    def set_input(self, img, label):
        self.img = img
        self.label = label
    def train(self):
        output = self.ST_model(self.img)
        loss = self.criterion_CE(output, self.label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return output, loss


    
        
    def inference(self):
        pass
    def save(self):
        pass
    def load(self):
        pass


    

