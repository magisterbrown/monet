import webdataset as wds
import cv2

class SimpleWds:
    def __init__(self, files: list):
        self.files = files


    def write(self, output: str):
        sink = wds.TarWriter(output, encoder=True)
        nm = 0
        for imgn in self.files:
            img = cv2.imread(imgn)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sink.write({
                '__key__': f'{nm}',
                'npy':img
            })
            nm+=1
        sink.close()
