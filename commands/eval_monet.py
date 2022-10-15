from .basic_sub import BasicSub
import pandas as pd
import numpy as np
import json
from monart.loss.fid import mfid

class MonetEval(BasicSub):
    def __init__(self,*args,**kwargs):
        arguments = [('-i', '--input', 'Input path to the zip submission file')]
        super().__init__(arguments = arguments, *args, **kwargs)
        df = pd.read_csv(self.args.input, index_col=0)
        self.monet = np.array(list(map(json.loads,(df['vectors'].tolist()))))

    def submit(self):
        batches = np.array_split(self.monet,2)
        print(mfid(batches[0], batches[1]))
