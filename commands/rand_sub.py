from .basic_sub import BasicSub
import numpy as np
import zipfile
import cv2

class RandSub(BasicSub):
    def __init__(self,*args,**kwargs):
        arguments = [('-e', '--elements', 'Elements to generate')]
        super().__init__(arguments = arguments, *args, **kwargs)

        self.elements = int(self.args.elements)
        self.imsize = 256
        self.zipf = zipfile.ZipFile(f'{self.resdr}images.zip', 'w', zipfile.ZIP_DEFLATED)

    def submit(self):
        for i in range(self.elements):
            img = np.random.randint(255, size=(self.imsize,self.imsize,3),dtype=np.uint8)
            _, buf = cv2.imencode('.jpg', img)
            self.zipf.writestr(f'{i}.jpg', buf)
    

