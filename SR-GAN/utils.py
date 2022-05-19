import os
import torchvision

def save_img(img, name):
    if not os.path.isdir("./results"):
        os.makedirs("./results")
        print("make directory ./results")
    torchvision.utils.save_image(img, "./results/{}.png".format(name))