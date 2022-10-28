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
        bucket = 'gs://monet-cool-gan/monet_wds.tar'
        device = xm.xla_device()
        self.trainer = XLAtrainer(bucket, device=device, save_pth=self.args.save)

    def submit(self):
        self.trainer.train()
