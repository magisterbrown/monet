from .base_command import BasicCommand
from monart.dswriters.wdswriters import SimpleWds
import glob

class TarCreate(BasicCommand):
    def __init__(self):
        self.add_arg('p','path','output file path')
        self.add_arg('l','list','dir with input files')
        super().__init__()
        ffolder = self.pathify(self.args.list)
        self.out = self.args.path

        self.wds = SimpleWds(glob.glob(f'{ffolder}/*'))

    def submit(self):
        self.wds.write(self.out) 


        
        


