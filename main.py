from commands.rand_sub import RandSub
from commands.monet_to_vec import MonetToVec
from commands.rate_submission import RateSub
from commands.pic_submission import PicSub


if __name__=='__main__':
    #rs = RandSub()
    #rs = MonetToVec()
    #rs = PicSub()
    rs = RateSub()
    rs.submit()
