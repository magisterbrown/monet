from .base_command import BasicCommand
from monart.training.generator import Generator, XLAGenerator

class Generate(BasicCommand):
    def __init__(self):
        self.add_arg('p','path','Path to the weights to load')
        self.add_arg('o','output','Path to the output folder')
        self.add_arg('m','mode','Mode x - xla p - normal')
        super().__init__()
        if(self.args.mode == 'p'):
            scales = [512]
            self.gen = Generator(scales, self.args.path)
        if(self.args.mode == 'x'):
            scales = []
            self.gen = XLAGenerator(scales, self.args.path)
        self.output = self.pathify(self.args.output)

    def submit(self):
        self.gen.generate(self.output)
