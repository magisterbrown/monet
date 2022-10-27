from commands.rand_sub import RandSub
from commands.monet_to_vec import MonetToVec
from commands.rate_submission import RateSub
from commands.pic_submission import PicSub
from commands.eval_monet import MonetEval
from commands.train_pgan import TrainPgan
from commands.gen_something import Generate
from commands.tarwr import TarCreate
from commands.xla_cm import TrainXLA


if __name__=='__main__':
    #rs = RandSub()
    #rs = MonetToVec()
    #rs = PicSub()
    #rs = RateSub()
    #rs = MonetEval()
    #rs = TrainPgan()
    #rs = Generate()
    #rs = TarCreate()
    rs = TrainXLA()
    rs.submit()
