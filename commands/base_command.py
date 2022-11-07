from argparse import ArgumentParser

class BasicCommand:
    arguments = list()
    def __init__(self):
        parser = ArgumentParser()
        for arg in self.arguments:
            parser.add_argument(arg[0], arg[1], help = arg[2])
        self.args = parser.parse_args()

    def add_arg(self, flag: str, name: str, comment: str):
        self.arguments.append((f'-{flag}',f'--{name}',comment))

    def submit(self):
        raise NotImplementedError

    @staticmethod
    def pathify(pth: str):
        if not pth.endswith('/'):
            pth += '/'
        return pth

class NewBasicCommand:
    arguments = list()
    def __init__(self, inpute):
        parser = ArgumentParser()
        for arg in self.arguments:
            parser.add_argument(arg[0], arg[1], help = arg[2])
        self.args = parser.parse_known_args(inpute)

    def add_arg(self, flag: str, name: str, comment: str):
        self.arguments.append((f'-{flag}',f'--{name}',comment))

    def submit(self):
        raise NotImplementedError

    @staticmethod
    def pathify(pth: str):
        if not pth.endswith('/'):
            pth += '/'
        return pth
