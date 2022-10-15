from .base_command import BasicCommand
from monart.training.generator import Generator

class Generate(BasicCommand):
    def __init__(self):
        self.add_arg('p','path','Path to the weights to load')
        self.add_arg('o','output','Path to the output folder')
        super().__init__()
        scales = [512]
        self.gen = Generator(scales, self.args.path)
        self.output = self.pathify(self.args.output)

    def submit(self):
        self.gen.generate(self.output)
