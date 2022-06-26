import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from main import build_args
from models.StarGANv2 import StarGANv2

if __name__=="__main__":
    args = build_args(is_test=True)
    #### setting ####
    load_path = "/media/data1/jeonghokim/GANs/StarGANv2/20220626/save_models/100000_100000.pth"
    #################
    model = StarGANv2(args)
    model.load(load_path)
    print(f"model is loaded from {load_path}")
    # latent-guided evaluation 
    latent_lpips_dict, latent_fid_dict, latent_msg = model.evaluate(mode="latent")
    # reference-guided evaluation
    ref_lpips_dict, ref_fid_dict, ref_msg = model.evaluate(mode="reference")
    msg = f"[Test]_[load path - {load_path}]"
    msg += latent_msg
    msg += ref_msg
    print(msg)