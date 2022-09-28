import torch
from pytorch_fid.inception import InceptionV3

def init_inception():
    
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    return model
