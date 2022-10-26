import torch
import glob
import webdataset as wds
import torch_xla.core.xla_model as xm
from models.gan_xla import XlaGan

class XLAtrainer:
    def __init__(self, bucket: str, device):
        self.bucket = bucket
        self.size = 4
        self.device = device
        self.init_model()

    def init_model(self):
        self.ganmodel = XlaGan(useGPU=False,
                             storeAVG=True,
                             device=self.device,
                             lambdaGP=10,
                             epsilonD=0.001)

    def get_loader(self, size: int):
        cessor = Allproc('npy', size)
        ds = wds.WebDataset(self.bucket).shuffle(30).decode().map(cessor.proc)
        loader = wds.WebLoader(ds, num_workers=4, batch_size=8)

        return loader

    def train(self, scales: list = [512, 512, 512, 512, 256, 128, 64, 32, 16]):
        scales = scales[:2]
        scales.append(None)
        dl = self.get_loader(self.size)
        epochs_at_step = [1,2,3]
        assert len(epochs_at_step)==len(scales)
        eps = iter(epochs_at_step)

        for sc in scales:
            for i in range(next(eps)):
                self.ganmodel = self.train_one_epoch(dl, self.ganmodel)
            if sc:
                self.ganmodel.addScale(sc)
                self.size*=2
                dl = self.get_loader(self.size)
    

    def train_one_epoch(self, dl, ganmodel):
        for key, data in enumerate(dl):
            data = data.to(device)
            losses = ganmodel.optimizeParameters()

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

































