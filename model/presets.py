import torch
import torchvision.transforms as TT
from pathlib import Path
import sys, os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import model.transforms as T


class SegTrainTransform:

    def __init__(self, size=(256,512), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([                                        
                                        T.ToTensor(),
                                        T.Resize(size),
                                        T.ColorJitter(hue=0.1, saturation=0.25),
                                        T.RandomHorizontalFlip(0.5),
                                        T.ConvertImageDtype(torch.float),
                                        T.Normalize(mean=mean, std=std),
                                        ]) 

    def __call__(self, img, mask):
        return self.transforms(img, mask) 

class SegEvalTransform:
    
    def __init__(self, size=(256,512), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([                                        
                                        T.ToTensor(),
                                        T.Resize(size),
                                        T.ConvertImageDtype(torch.float),
                                        T.Normalize(mean=mean, std=std),
                                        ])

    def __call__(self, img, mask):
        return self.transforms(img, mask) 

class SegInferTransform:
    def __init__(self, size=(256,512), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = TT.Compose([                                        
                                        TT.ToTensor(),
                                        TT.Resize(size, interpolation=TT.InterpolationMode.NEAREST),
                                        TT.ConvertImageDtype(torch.float),
                                        TT.Normalize(mean=mean, std=std),
                                        ])

    def __call__(self, img):
        return self.transforms(img) 
