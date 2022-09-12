# Fruit Ninja using Hands
A clone of the fruit ninja game, where the player uses their hands to slash the fruits

For demos and details on model training, please visit the Project Page: [https://www.aliasghar.tech/home/fruit-ninja/](https://www.aliasghar.tech/home/fruit-ninja/)

## Instructions on how to play the game
### 1. Requirements
1. For now, you must have a CUDA compatible device in order to play the game
2. [Install PyTorch](https://pytorch.org/) based on your preferences
3. Install Pygame
`pip install pygame`
4. Download the model weights: [traced_hand_segmentation_model.pt](https://drive.google.com/file/d/1W0GzBs83SyRIafPt4Oq8hbO3zuo0ubFb/view?usp=sharing)

### 2. Run the game
1. Clone this repository `git clone https://github.com/aliasghar53/fruit-ninja-with-hands.git`
2. Go to game directory `cd game`
3. Run the game indicating where you have stored the model file `python3 game.py --path /path/to/model_weights`

## Issues
Please use the Issues tab to share your questions/concerns

## Acknowledgements
- GetProject-org for their [Fruit Ninja replica](https://github.com/GetProjects-org/Fruit-Ninja-Game-in-Python)
- [Pytorch's reference code for training segmentation models](https://github.com/pytorch/vision/tree/main/references/segmentation)
