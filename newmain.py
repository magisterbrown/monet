import sys

class Boss:
    def __init__(self):
        self.els = dict()

    def add_command(self, key: str, func: callable):
        self.els[key] = func

    def get_command(self, key):
        return self.els[key]()

def train_mul_xla():
        from commands.xla_cm import TrainMultiXLA as comandor
        return comandor

if __name__=='__main__':
    boss = Boss()
    boss.add_command('train_mul_xla', train_mul_xla)
    
    try:
        subcommand = sys.argv[1]

        Comador = boss.get_command(subcommand) 
        task = Comador()
        task.submit()
    except IndexError:
        print('Options: ')
        print(list(boss.els.keys()))  # Display help if no arguments were given.


    



