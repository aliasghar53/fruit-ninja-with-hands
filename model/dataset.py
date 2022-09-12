from torch.utils.data import Dataset
from pymatreader import read_mat
import glob
import numpy as np
import cv2
import torch
from PIL import Image

from pathlib import Path
import sys, os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import model.transforms as T
from model.presets import SegEvalTransform, SegTrainTransform, SegInferTransform

class EgoHands(Dataset):
    def __init__(self, ROOT="./data/egohands_data/", transform=None, mode="eval", multi_class=False, size=(256,512)):
        self.label_data = read_mat(ROOT + 'metadata.mat')

        self.frame_paths = sorted(glob.glob(ROOT + "_LABELLED_SAMPLES/*/*.jpg"))
        self.masks = []

        for video_index in range(48):
            for frame_num in range(100):

                mask =  np.zeros((720,1280),np.uint8)

                for i, cls in enumerate(['myleft', 'myright', 'yourleft', 'yourright'], start=1):

                    label = self.label_data["video"]["labelled_frames"][video_index][cls][frame_num]

                    if label.size > 0:
                        cv2.fillPoly(mask, pts=np.int32([label]), color=i if multi_class else 1)
                    
                self.masks.append(mask)
        
        if (mode == "train") and (transform is None):
            self.transform = SegTrainTransform(size=size)
        elif (mode == "eval") and (transform is None):
            self.transform = SegEvalTransform(size=size)
        else:
            self.transform = transform  

    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.frame_paths[index]).convert("RGB")
        mask = self.masks[index]

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask


class FreiHands(Dataset):
    def __init__(self, ROOT="./data/FreiHAND_pub_v2_eval/evaluation/", mode="eval", transform=None, size=224):
        self.img_paths = sorted(glob.glob(ROOT + "rgb/*.jpg"))
        self.mask_paths = sorted(glob.glob(ROOT + "segmap/*.png"))

        if (mode == "train") and (transform is None):
            self.transform = SegTrainTransform(size=size)
        elif (mode == "eval") and (transform is None):
            self.transform = SegEvalTransform(size=size)
        else:
            self.transform = transform  
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        
        img = Image.open(self.img_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index])
        mask = np.array(mask)
        mask = np.where(mask > 0, 1, 0)

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

class EgoHandsTest(Dataset):
    def __init__(self, ROOT="./data/egohands_data/", transform=None, mode="test", size=(256,512)):
        self.frame_paths = sorted(glob.glob(ROOT + "_LABELLED_SAMPLES/*/*.jpg"))
        
        if (mode == "test") and (transform is None):
            self.transform = SegInferTransform(size=size)
        else:
            self.transform = transform  

    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.frame_paths[index]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img

# Some testing of the code
if __name__ == "__main__":
    obj = FreiHands(size=224)
    
    img, mask = obj[0]

    img = img.numpy().transpose((1,2,0))
    mask = mask.numpy()


    # c1 = np.array([255,0,0], dtype='int8')
    c2 = np.array([0,255,0], dtype='int8')
    # c3 = np.array([0,0,255], dtype='int8')
    # c4 = np.array([120,120,120], dtype='int8')

    masked_img = np.where(mask[...,None] == 1, c2, img)
    # masked_img = np.where(mask[...,None] == 2, c2, masked_img)
    # masked_img = np.where(mask[...,None] == 3, c3, masked_img)
    # masked_img = np.where(mask[...,None] == 4, c4, masked_img)
    
    masked_img = Image.fromarray(np.uint8(masked_img))
    masked_img.show()
