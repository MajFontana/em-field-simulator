import pygame
import numpy




import os
if os.name == "nt":
    import ctypes
    ctypes.windll.user32.SetProcessDPIAware()

    



class ArrayRenderer:
    def pygameSurface(array, size):
        raw = numpy.clip(array * 255, 0, 255).astype("uint8")
        if len(raw.shape) == 2:
            rendered = pygame.surfarray.make_surface(numpy.repeat(raw[..., None], 3, 2))
        elif raw.shape[2] == 3:
            rendered = pygame.surfarray.make_surface(raw)
        elif raw.shape[2] == 4:
            rendered = pygame.Surface(raw.shape[:2], pygame.SRCALPHA)
            pygame.surfarray.blit_array(rendered, raw[..., :3])
            alpha = numpy.array(rendered.get_view("A"), copy=False)
            alpha[:] = raw[..., 3]
        else:
            raise AttributeError("Invalid array shape")
        surf = pygame.transform.scale(rendered, size)
        return surf




class Window:
    def __init__(self, size):
        self.size = size
        pygame.init()
        self.screen = pygame.display.set_mode(size, pygame.SRCALPHA)
        self.clock = pygame.time.Clock()

    def update(self):
        cont = True
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cont = False
        return cont

    def close(self):
        pygame.quit()

    def drawArray(self, array, size, position):
        pxsize = [int(self.size[i] * size[i]) for i in range(2)]
        pxpos = [int(self.size[i] * position[i]) for i in range(2)]
        surf = ArrayRenderer.pygameSurface(array, pxsize)
        self.screen.blit(surf, pxpos)

    def fill(self, color):
        self.screen.fill(color)

    def sleep(self, fps):
        self.clock.tick(fps)