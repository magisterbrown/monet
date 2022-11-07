from .base_command import BasicCommand
from monart.training.xla_trainer import XLAtrainer
from monart.copy_env import copy_process
import torch_xla.core.xla_model as xm
import os


class TrainXLA(BasicCommand):
    def __init__(self):
        self.add_arg('p','pid','process id to copy')
        self.add_arg('s','save','path to save pth')
        super().__init__()
        copy_process(self.args.pid)
        self.bucket = 'gs://monet-cool-gan/'
        archive = f'{self.bucket}monet_wds.tar'
        device = xm.xla_device()
        self.trainer = XLAtrainer(archive, device=device, save_pth=self.args.save)

    def submit(self):
        self.trainer.train()
        os.system(f'gsutil cp {self.args.save} {self.bucket}')

class TrainMultiXLA():
    def submit(self):
        print('submit')
    
