from argparse import ArgumentParser
import zipfile

class BasicSub:
    def __init__(self, arguments=list()):
        parser = ArgumentParser()
        arguments.append(('-o','--output', 'Path to output'))
        for arg in arguments:
            parser.add_argument(arg[0], arg[1], help = arg[2])
        self.args = parser.parse_args()

        self.resdr = self.args.output
        self.zipf = zipfile.ZipFile(f'{self.resdr}/images.zip', 'w', zipfile.ZIP_DEFLATED)

    def submit(self):
        raise NotImplementedError
