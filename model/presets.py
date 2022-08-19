import torch
import transforms as T
import torchvision.transforms as TT

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
                                        # TT.Resize(size),
                                        TT.ConvertImageDtype(torch.float),
                                        TT.Normalize(mean=mean, std=std),
                                        ])

    def __call__(self, img):
        return self.transforms(img) 