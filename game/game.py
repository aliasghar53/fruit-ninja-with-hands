import pygame
import os
import random
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
from camera import Webcam

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from model.presets import SegInferTransform


# transform for inference
transform = SegInferTransform(size=224)

# build and load model weights
model = deeplabv3_resnet50(pretrained=False, progress=True, aux_loss=False)
model.classifier = DeepLabHead(in_channels = 2048, num_classes = 2)
model_state_dict = torch.load("../model/ckpt/ckpt_bb/best.pth")["model"]
model.load_state_dict(model_state_dict, strict=True)
model.eval()


# check if gpu is available
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "Sorry, the game won't work well without an NVIDIA GPU at the moment"
model.to(device)

# used for making hand masks
c = np.array([0,255,0], dtype='uint8')

player_lives = 3                                                #keep track of lives
score = 0                                                       #keeps track of score
fruits = ['melon', 'orange', 'pomegranate', 'guava', 'bomb']    #entities in the game

# initialize pygame and create window
WIDTH = 640
HEIGHT = 480
FPS = 20                                                 #controls how often the gameDisplay should refresh. In our case, it will refresh every 1/5th second
pygame.init()
pygame.display.set_caption('Fruit-Ninja Game in Python -- GetProjects.org')
gameDisplay = pygame.display.set_mode((WIDTH, HEIGHT))   #setting game display size
webcam = Webcam((WIDTH,HEIGHT), gameDisplay)
clock = pygame.time.Clock()

# Define colors
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

background = pygame.image.load('back.jpg')                                  #game background
font = pygame.font.Font(os.path.join(os.getcwd(), 'comic.ttf'), 42)
score_text = font.render('Score : ' + str(score), True, (255, 255, 255))    #score display
lives_icon = pygame.image.load('images/white_lives.png')                    #images that shows remaining lives

# generate hand mask
def get_hand_mask(img_surface, device):
    '''
    take a image (pygame Surface) and return a mask (pygame Mask) of the hand
    '''

    # convert image to np array
    img = pygame.surfarray.array3d(img_surface)

    # transform and reshape
    input = transform(img)    
    input_batch = input.unsqueeze(0)

    # perform inference
    input_batch = input_batch.to(device)

    with torch.inference_mode():
        output = model(input_batch)['out'][0]

    # post processing
    output_predictions = output.argmax(axis=0)

    output_predictions=output_predictions.detach().cpu()  
    output_predictions = F.resize(output_predictions.unsqueeze(0), (WIDTH,HEIGHT), interpolation=T.InterpolationMode.NEAREST).squeeze(0)
    output_predictions = output_predictions.numpy()
    
    masked_img = np.where(output_predictions[...,None] == 1, c, 0)

    masked_img = pygame.surfarray.make_surface(masked_img)

    masked_img.set_colorkey(c)
    mask = pygame.mask.from_surface(masked_img)
    mask.invert()
    mask = mask.connected_component()

    return mask

# Generalized structure of the fruit Dictionary
def generate_random_fruits(fruit):
    fruit_path = "images/" + fruit + ".png"
    image = pygame.image.load(fruit_path)
    data[fruit] = {
        'img': image,
        'x' : random.randint(100,WIDTH-100),          #where the fruit should be positioned on x-coordinate
        'y' : HEIGHT+20,
        'speed_x': random.randint(-10,10),      #how fast the fruit should move in x direction. Controls the diagonal movement of fruits
        'speed_y': random.randint(-300, -200),    #control the speed of fruits in y-directionn ( UP )
        'throw': False,                         #determines if the generated coordinate of the fruits is outside the gameDisplay or not. If outside, then it will be discarded
        't': 0,                                 #manages the flight time
        'hit': False,
        "mask": pygame.mask.from_surface(image)
    }

    if random.random() >= 0.75:     #Return the next random floating point number in the range [0.0, 1.0) to keep the fruits inside the gameDisplay
        data[fruit]['throw'] = True
    else:
        data[fruit]['throw'] = False

# Dictionary to hold the data the random fruit generation
data = {}
for fruit in fruits:
    generate_random_fruits(fruit)

def hide_cross_lives(x, y):
    gameDisplay.blit(pygame.image.load("images/red_lives.png"), (x, y))

# Generic method to draw fonts on the screen
font_name = pygame.font.match_font('comic.ttf')
def draw_text(display, text, size, x, y):
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, WHITE)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    display.blit(text_surface, text_rect)

# draw players lives
def draw_lives(display, x, y, lives, image) :
    for i in range(lives) :
        img = pygame.image.load(image)
        img_rect = img.get_rect()       #gets the (x,y) coordinates of the cross icons (lives on the the top rightmost side)
        img_rect.x = int(x + 35 * i)    #sets the next cross icon 35pixels awt from the previous one
        img_rect.y = y                  #takes care of how many pixels the cross icon should be positioned from top of the screen
        display.blit(img, img_rect)

# show game over display & front display
def show_gameover_screen():
    gameDisplay.blit(background, (0,0))
    draw_text(gameDisplay, "FRUIT NINJA!", 90, WIDTH / 2, HEIGHT / 4)
    if not game_over :
        draw_text(gameDisplay,"Score : " + str(score), 50, WIDTH / 2, HEIGHT /2)

    draw_text(gameDisplay, "Press a key to begin!", 64, WIDTH / 2, HEIGHT * 3 / 4)
    pygame.display.flip()
    waiting = True
    while waiting:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYUP:
                waiting = False

# Game Loop
first_round = True
game_over = True        #terminates the game While loop if more than 3-Bombs are cut
game_running = True   
dt = 0  #used to manage the game loop
max_overlap = 0
while game_running :
    if game_over :
        if first_round :
            show_gameover_screen()
            first_round = False
        game_over = False
        player_lives = 3
        draw_lives(gameDisplay, WIDTH-100, 5, player_lives, 'images/red_lives.png')
        score = 0

    for event in pygame.event.get():
        # checking for closing window
        if event.type == pygame.QUIT:
            webcam.cam.stop()
            game_running = False
            

    frame = webcam.get_frame() 
    hand_mask = get_hand_mask(frame, device)
    
    # if hand_mask.count()>10000:
    #     coords = hand_mask.outline(every=5)
    #     pygame.gfxdraw.filled_polygon(frame, coords, (0,255,0))
        
    gameDisplay.blit(frame, (0,0))
    gameDisplay.blit(score_text, (0, 0))
    draw_lives(gameDisplay, WIDTH-110, 5, player_lives, 'images/red_lives.png')

    for key, value in data.items():
        if value['throw']:
            value['x'] += value['speed_x'] * (dt/1000)          #moving the fruits in x-coordinates
            value['y'] += value['speed_y'] * (dt/1000)         #moving the fruits in y-coordinate
            value['speed_y'] += (100 * dt/1000)    #increasing y-corrdinate
            # value['t'] += 1                         #increasing speed_y for next loop

            if value['y'] <= HEIGHT+20:
                gameDisplay.blit(value['img'], (value['x'], value['y']))    #displaying the fruit inside screen dynamically
                hand_rect = hand_mask.get_rect()
                overlap_mask = hand_mask.overlap_mask(value['mask'], (value['x'] - hand_rect.left, value["y"] - hand_rect.top))
                
            else:
                generate_random_fruits(key)        
 
            if overlap_mask.count() > 500 and not value["hit"] and hand_mask.count()>10000:
                
                gameDisplay.blit(overlap_mask.to_surface(setcolor=(255,0,255,255), unsetcolor=(0,0,0,0)), (hand_rect.left, hand_rect.top))
                if key == 'bomb':
                    player_lives -= 1
                    if player_lives == 0 :
                        show_gameover_screen()
                        game_over = True

                    half_fruit_path = "images/explosion.png"
                else:
                    half_fruit_path = "images/" + "half_" + key + ".png"

                value['img'] = pygame.image.load(half_fruit_path)
                value['speed_x'] += 10
                if key != 'bomb' :
                    score += 1
                score_text = font.render('Score : ' + str(score), True, (255, 255, 255))
                value['hit'] = True
        else:
            generate_random_fruits(key)

    pygame.display.update()
    dt = clock.tick(FPS)      # keep loop running at the right framerate 
    print(1000/dt) 

pygame.quit()
