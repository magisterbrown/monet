from .base_command import BasicCommand
from monart.training.trainer import Trainer
import glob
import torch
from submodules.pytorch_GAN_zoo_xla.models.progressive_gan import ProgressiveGAN

class TrainPgan(BasicCommand):
    def __init__(self):
        self.add_arg('p','path','Path to the input data')
        self.add_arg('s','save','Path to save the model')
        super().__init__()
        path = self.pathify(self.args.path)
        gl = glob.glob(f'{path}*')
        self.trainer = Trainer(gl[0])

    def submit(self):
        self.trainer.train() 
        model = self.trainer.get_generator()
        torch.save(model.state_dict(), self.args.save)
