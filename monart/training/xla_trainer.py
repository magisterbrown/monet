import torch
import glob
import time
import webdataset as wds
import torch_xla.core.xla_model as xm
from models.gan_xla import XlaGan
import cv2
import numpy as np
import time

        

class XLAtrainer:
    def __init__(self, bucket: str, device, save_pth: str):
        self.bucket = bucket
        self.save_pth = save_pth
        self.size = 4
        self.device = device
        self.init_model()

    def init_model(self):
        self.ganmodel = XlaGan(useGPU=False,
                             storeAVG=True,
                             device=self.device,
                             lambdaGP=10,
                             epsilonD=0.001)
        self.ganmodel.updateSolversDevice()

    def get_loader(self, size: int):
        cessor = Allproc('npy', size)
        ds = wds.WebDataset(self.bucket).shuffle(30).decode().map(cessor.proc)
        loader = wds.WebLoader(ds, num_workers=4, batch_size=8)

        return loader

    def train(self, scales: list = [512, 512, 512, 512, 256, 128, 64, 32, 16]):
        scales = [512, 512, 512, 512, 256]
        scales.append(None)
        dl = self.get_loader(self.size)
        epochs_at_step = [7,11,11,14,14,17]
        assert len(epochs_at_step)>=len(scales)-1
        eps = iter(epochs_at_step)

        for key, sc in enumerate(scales):
            for i in range(next(eps)):
                stt = time.time()
                self.ganmodel = self.train_one_epoch(dl, self.ganmodel)
                print(f'EPoch {i} scale {key} time {time.time()-stt}')
            if sc:
                self.ganmodel.addScale(sc)
                self.size*=2
                dl = self.get_loader(self.size)
        xm.save(self.ganmodel.netG.state_dict(), self.save_pth)
    

    def train_one_epoch(self, dl, ganmodel):
        rl = 0
        fins = dict()
        for key, data in enumerate(dl):
            st = time.time()
            data = data.to(self.device)
            losses = ganmodel.optimizeParameters(data)
            for k,v in losses.items():
                try:
                    fins[k].append(v)
                except:
                    fins[k] = [v]
            if(key>rl):
                rl = key
            if(key<1):
                print(f'ST: {key}/{rl} secs: {time.time()-st}')

        for k,v in fins.items():
            print(f'{k} {np.mean(v)}')

        return ganmodel


class Allproc:
    def __init__(self, key: str, side: int):
        self.key = key
        self.side = side

    def proc(self, el):
        el = el[self.key]
        el = cv2.resize(el, (self.side,self.side))
        el = np.moveaxis(el,2,0).astype(np.float32)
        el = (el/127.5)-1
        return el

































