import torch
from submodules.pytorch_GAN_zoo_xla.models.progressive_gan import ProgressiveGAN
from models.gan_xla import XlaGan
from visualization.visualizer import resizeTensor
import matplotlib.pyplot as plt
import numpy as np
import cv2

class Generator:
    def __init__(self, depth_scales: list, weights_pth: str, res_size=64):

        self.gan_init()
        for sc in depth_scales:
            self.proggan.addScale(sc)
        self.proggan.netG.load_state_dict(torch.load(weights_pth))
        self.res_size = res_size

    def gan_init(self):
        self.proggan = ProgressiveGAN(useGPU=True,
                             storeAVG=True) 

    def generate(self, path: str):
        noiseData, _ = self.proggan.buildNoiseData(1)
        res = self.proggan.test(noiseData,getAvG=False)
        res = resizeTensor(res,(self.res_size, self.res_size))
        im0 = np.moveaxis(res[0].numpy(),[0,1,2], [2,0,1])
        cv2.imwrite(f'{path}im0.jpg', im0*255)

class XLAGenerator(Generator):

    def gan_init(self):
        self.proggan = XlaGan(useGPU=False,
                             storeAVG=True,
                             lambdaGP=10,
                             epsilonD=0.001)

