import pygame
import pygame.camera
from pygame.locals import *

class Webcam:
    def __init__(self, size, display):
        self.size = size
        self.display = display

        pygame.camera.init()

        # verify if camera is available
        self.clist = pygame.camera.list_cameras()
        if not self.clist:
            raise ValueError("Sorry, no cameras detected.")
        self.cam = pygame.camera.Camera(self.clist[0], self.size)
        self.cam.start()

        # create a surface to capture to.  for performance purposes
        # bit depth is the same as that of the display surface.
        self.snapshot = pygame.surface.Surface(self.size, 0, self.display)

    def get_frame(self):
        # To not tie the framerate to the camera, check
        # if the camera has an image ready.  note that while this works
        # on most cameras, some will never return true.
        if self.cam.query_image():
            self.snapshot = self.cam.get_image(self.snapshot)

        # flip and return
        self.flipped = pygame.transform.flip(self.snapshot, True, False)
        
        return self.flipped