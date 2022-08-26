import pygame
import pygame.camera
from pygame.locals import *

class Webcam:
    def __init__(self, size, display):
        self.size = size
        self.display = display

        pygame.camera.init()

        # this is the same as what we saw before
        self.clist = pygame.camera.list_cameras()
        if not self.clist:
            raise ValueError("Sorry, no cameras detected.")
        self.cam = pygame.camera.Camera(self.clist[0], self.size)
        self.cam.start()

        # create a surface to capture to.  for performance purposes
        # bit depth is the same as that of the display surface.
        self.snapshot = pygame.surface.Surface(self.size, 0, self.display)

    def get_frame(self):
        # if you don't want to tie the framerate to the camera, you can check
        # if the camera has an image ready.  note that while this works
        # on most cameras, some will never return true.
        if self.cam.query_image():
            self.snapshot = self.cam.get_image(self.snapshot)

        # blit it to the display surface.  simple!
        self.flipped = pygame.transform.flip(self.snapshot, True, False)
        # self.display.blit(self.flipped , (0,0))
        # pygame.display.flip()
        return self.flipped