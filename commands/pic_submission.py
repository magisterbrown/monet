from .basic_sub import BasicSub
import matplotlib.pyplot as plt 
import zipfile
import random
import os
import cv2

class PicSub(BasicSub):
    def __init__(self):
        arguments = [('-i', '--input', 'Input path to the dir with images'),
        ('-e', '--elements', 'Elements to generate')]
        super().__init__(arguments = arguments)
        self.inputs = self.pathify(self.args.input)
        self.elements = int(self.args.elements)

        self.zipf = zipfile.ZipFile(f'{self.resdr}images.zip', 'w', zipfile.ZIP_DEFLATED)

    def submit(self):
        files = os.listdir(self.inputs)

        for i in range(self.elements):
            fl = random.choice(files) 
            img = cv2.imread(f'{self.inputs}{fl}')
            _, buf = cv2.imencode('.jpg', img)
            self.zipf.writestr(f'{i}.jpg', buf)

