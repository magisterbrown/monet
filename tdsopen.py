import torchdata
from torchdata.datapipes.iter import FileOpener, FileLister
import torchdata.datapipes.iter as pipes
from torchdata.datapipes.utils import StreamWrapper
from webdatasett import webdataset as wds
import glob
import cv2
import numpy as np
#wdsp = torchdata.datapipes.iter.WebDataset("data/monet_wds.tar")
camvid_itertable = pipes.IterableWrapper(["data/monet_wds.tar"])
# datapipe1 = FileLister("data/", "monet_wds.tar")
def rh(key, val):
    if(isinstance(val, StreamWrapper)):
        val = val.read()
    return wds.Continue(key.strip('.'),val)
decoder = wds.Decoder([rh])
def muped(x):
    print(x)
    return x
wdsp = FileOpener(camvid_itertable, mode="b").load_from_tar().webdataset().map(muped)
itwdsp = iter(wdsp)
rese = next(itwdsp)
print(rese)
ndd = dict()
ndd['__key__'] = rese['__key__']
ndd['img.npy'] = rese['.img.npy']
print(ndd)
decoder.decode(ndd)
