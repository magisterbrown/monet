from .basic_sub import BasicSub
import cv2
import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
from monart.process.est_ception import process_batch
from monart.models.inception import init_inception
from monart.process.est_ception import run_ten_model
from monart.loss.fid import mfid
import torch
import json

class RateSub(BasicSub):
    def __init__(self):
        arguments = [('-i', '--input', 'Input path to the zip submission file'),
                    ('-m', '--monet', 'Input path to the monet vectors df csv'),
                    ('-b', '--bs', 'Batch size')]
        super().__init__(arguments = arguments)
        inputt = self.args.input
        self.bs = 16
        if self.args.bs:
            self.bs = self.args.bs
        df = pd.read_csv(self.args.monet, index_col=0)
        self.archive = zipfile.ZipFile(inputt, 'r')
        self.model = init_inception()
        self.monet = np.array(list(map(json.loads,(df['vectors'].tolist()))))

    def submit(self):
        files = self.archive.infolist()
        batches = np.array_split(files,len(files)//16)
        finvecs = list()
        for batch in batches:
            loaded = list()
            for fl in batch:
                data = self.archive.read(fl)
                img = cv2.imdecode(np.frombuffer(data, np.uint8),1)
                loaded.append(img)
            ten = process_batch(loaded)
            res = run_ten_model(ten, self.model)
            finvecs.append(res)
        
        finres = np.concatenate(finvecs)
        print(mfid(self.monet,finres))



