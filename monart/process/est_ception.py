import torch
import numpy as np
import cv2

def read_batch(batch: np.array, path: str):
    ims = list()
    for im in batch:
        im_cv = cv2.imread(f'{path}{im}',cv2.IMREAD_COLOR)
        ims.append(im_cv)

    return ims

def process_batch(batch: list):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    ims = list()
    for im in batch:
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ims.append(im_rgb)
    ims = np.stack(ims)
    ims = np.transpose(ims, (0,3,1,2)).astype(np.float32)/255
    ten = torch.from_numpy(ims).to(device)

    return ten

def run_ten_model(ten, model):
    with torch.no_grad():
        res = model(ten)
    sqz = np.squeeze(res[0].cpu().numpy())
    return sqz

def load_batch(batch: np.array, path: str, model):

    batch = read_batch(batch, path)
    ten = process_batch(batch)
    res = run_ten_model(ten, model)

    return res

