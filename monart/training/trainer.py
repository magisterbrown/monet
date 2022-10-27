from monart.datasets.mockds import MockDatases, MockImageDataset
import torch
import glob
from models.progressive_gan import ProgressiveGAN
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, pth_ds: str):
        self.ds = MockImageDataset(pth_ds, 4800, 4)
        self.init_model()
        
    def init_model(self):
        self.ganmodel = ProgressiveGAN(useGPU=False,
                             storeAVG=True,
                             lambdaGP=0,#10,
                             epsilonD=0)#0.001)
        self.ganmodel.updateSolversDevice()

    def get_generator(self):
        return self.ganmodel.netG

    def train(self):
        scales = [512]
        scales.append(None)
        for sc in scales:
            dl = DataLoader(self.ds, batch_size=4, shuffle=True)
            for i in range(1):
                self.ganmodel = self.train_one_epoch(dl, self.ganmodel)
            if sc:
                self.ganmodel.addScale(sc)
                self.ds.set_img_side(self.ds.img_side*2)
    

    def train_one_epoch(self, dl, ganmodel):
        for key, data in enumerate(dl):
            losses = ganmodel.optimizeParameters(data.to(torch.float32))
            if key%30==29:
                print(key)

        return ganmodel
