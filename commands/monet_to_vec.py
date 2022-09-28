from .basic_sub import BasicSub
import numpy as np
import cv2
import os
import pandas as pd

from monart.process.est_ception import load_batch
from monart.models.inception import init_inception

class MonetToVec(BasicSub):
    def __init__(self):
        arguments = [('-i', '--input', 'Input path folder with images')]
        super().__init__(arguments = arguments)

        self.input_p = self.pathify(self.args.input)
        self.bs = 16
        self.model = init_inception()

    def submit(self):
        images = os.listdir(self.input_p)
        batches = np.array_split(np.array(images),len(images)//self.bs)
        dfs = list()

        for b in batches:
            sqz = load_batch(b, self.input_p, self.model)
            data = {'names': b.tolist(), 'vectors': sqz.tolist()}
            df = pd.DataFrame(data)
            dfs.append(df) 

        alldf = pd.concat(dfs)
        alldf.to_csv(f'{self.resdr}monvec.csv')
