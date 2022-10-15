from torch.utils.data import Dataset
import cv2
import numpy as np

class MockDatases(Dataset):
    def __init__(self, size: int, img_side: int):
        self.size = size
        self.img_side = img_side

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return np.random.normal(size=(3,self.img_side,self.img_side)) 

class MockImageDataset(MockDatases):
    def __init__(self, im_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.img_or = cv2.imread(im_path)
        self.set_img_side(self.img_side)

    def __getitem__(self, idx):
        img = np.moveaxis(self.img, [0, 1, 2], [2,1,0]) 
        img = self.normalize(img)

        return img
    
    def normalize(self, x):
        x = x/255
        x-=0.5
        x/=0.5

        return x

    def set_img_side(self, x):
        self.img_side = x
        self.img = cv2.resize(self.img_or, (self.img_side, self.img_side), interpolation = cv2.INTER_AREA)

    def get_disp(self, idx):
        return self.img
