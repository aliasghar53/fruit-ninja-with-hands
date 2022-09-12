from camera import Webcam
import pygame
from pygame.locals import *
import pygame.gfxdraw
from pathlib import Path
import sys
import os
import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import numpy as np
import torchvision.transforms as T
from torchvision.transforms import functional as F
import tensorrt
import torch_tensorrt


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from model.presets import SegInferTransform


# Set up pygame
size = (640,480)
display = pygame.display.set_mode(size, 0)
webcam = Webcam(size, display)

# transform for inference
transform = SegInferTransform(size=(224))

# build and load model weights
# model = deeplabv3_resnet50(progress=True, aux_loss=False)
# model.classifier = DeepLabHead(in_channels = 2048, num_classes = 2)
# model_state_dict = torch.load("../model/ckpt/ckpt_bb/best.pth")["model"]
# model.load_state_dict(model_state_dict, strict=True)
model = torch.jit.load("/home/ali/fruit_ninja/game/traced_model.pt")
model.eval()

# check if gpu is available
if torch.cuda.is_available():
    model.to('cuda')

# used for making hand masks
c = np.array([0,255,0], dtype='uint8')

def get_hand_mask(img_surface):
    '''
    take a image (pygame Surface) and return a mask (pygame Mask) of the hand
    '''

    # convert image to np array
    img = pygame.surfarray.array3d(img_surface)

    # transform and reshape
    input = transform(img)    
    input_batch = input.unsqueeze(0)

    # perform inference
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')

    with torch.inference_mode():
        output = model(input_batch)['out'][0]

    # post processing
    output_predictions = output.argmax(axis=0)

    output_predictions=output_predictions.detach().cpu()  
    output_predictions = F.resize(output_predictions.unsqueeze(0), (640,480), interpolation=T.InterpolationMode.NEAREST).squeeze(0)
    output_predictions = output_predictions.numpy()
    
    masked_img = np.where(output_predictions[...,None] == 1, c, 0)

    masked_img = pygame.surfarray.make_surface(masked_img)

    masked_img.set_colorkey(c)
    mask = pygame.mask.from_surface(masked_img)
    mask.invert()
    mask = mask.connected_component()

    return mask
    

clock = pygame.time.Clock()

going = True
while going:
    events = pygame.event.get()
    for e in events:
        if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
            # close the camera safely
            webcam.cam.stop()
            going = False

    frame = webcam.get_frame() 
    mask = get_hand_mask(frame)
    
    if mask.count()>10000:
        coords = mask.outline(every=5)
        pygame.gfxdraw.filled_polygon(frame, coords, (0,255,0, 120))
        
    display.blit(frame, (0,0))
    pygame.display.flip()
    clock.tick(30)
    print(clock.get_fps())