from torch.utils.data import Dataset
from pymatreader import read_mat
import glob
import numpy as np
import cv2
import transforms as T
import torch
from PIL import Image

class EgoHands(Dataset):
    def __init__(self, ROOT="./egohands_data/", transform=None, mode="train", multi_class=False):
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
            self.transform = T.Compose([
                                        T.Resize((256,512)),
                                        T.ToTensor(),
                                        T.ColorJitter(hue=0.1, saturation=0.25),
                                        T.RandomHorizontalFlip(0.5),
                                        T.ConvertImageDtype(torch.float),
                                        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                        ])
        elif (mode == "eval") and (transform is None):
            self.transform = T.Compose([
                                        T.Resize((256,512)),
                                        T.ToTensor(),
                                        T.ConvertImageDtype(torch.float),
                                        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                        ])
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


# Some testing of the code
if __name__ == "__main__":
    obj = EgoHands(mode="eval")

    img, mask = obj[1710]

    print(torch.max(mask).item())

    
    img = img.numpy().transpose((1,2,0))
    mask = mask.numpy()


    c1 = np.array([255,0,0], dtype='int8')
    c2 = np.array([0,255,0], dtype='int8')
    c3 = np.array([0,0,255], dtype='int8')
    c4 = np.array([120,120,120], dtype='int8')

    masked_img = np.where(mask[...,None] == 1, c4, img)
    masked_img = np.where(mask[...,None] == 2, c2, masked_img)
    masked_img = np.where(mask[...,None] == 3, c3, masked_img)
    masked_img = np.where(mask[...,None] == 4, c4, masked_img)
    
    masked_img = Image.fromarray(np.uint8(masked_img))
    masked_img.show()
