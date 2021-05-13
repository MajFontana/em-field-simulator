import numpy
import pygame
import math




COULOMB_CONST = 8987551792.3
LIGHT_SPEED = 299792458




class EField:
    def __init__(self, size, scale, timestep):
        self.size = size
        self.scale = scale
        abssize = [(dim - 1) * scale for dim in size]

        self.timestep = timestep
        dstep = timestep * LIGHT_SPEED
        radii = list(numpy.arange(0, math.hypot(math.hypot(abssize[0], abssize[1]), abssize[2]) + dstep, dstep))

        axes = [numpy.linspace(-abssize[i], abssize[i], size[i] * 2 - 1) if size[i] > 1 else numpy.array([0]) for i in range(3)]
        coords = numpy.stack(numpy.meshgrid(*axes, indexing="ij"), axis=-1)
        self.modelcenter = tuple([int(dim / 2) for dim in coords.shape[:-1]])
        
        distsquared = numpy.einsum("xyzp,xyzp->xyz", coords, coords)
        dist = numpy.sqrt(distsquared)
        with numpy.errstate(divide="ignore", invalid="ignore"):
            unitfield = (COULOMB_CONST / distsquared)[..., None] * (coords / dist[..., None])
        unitfield[self.modelcenter] = 0
        self.model = numpy.array([(dist > radii[i]) & (dist < radii[i + 1]) for i in range(len(radii) - 1)])[..., None] * unitfield[None,...]

        self.field = numpy.zeros((len(radii) - 1, *size, 3))
        self.chargefield = numpy.zeros(size)
        self.time = 0

    def update(self):
        self.field = numpy.roll(self.field, -1, 0)
        self.field[-1] = 0
        for x in range(self.size[0]):
            xstart = self.modelcenter[0] - x
            xstop = xstart + self.size[0]
            for y in range(self.size[1]):
                ystart = self.modelcenter[1] - y
                ystop = ystart + self.size[1]
                for z in range(self.size[2]):
                    charge = self.chargefield[x, y, z]
                    if charge:
                        zstart = self.modelcenter[2] - z
                        zstop = zstart + self.size[2]
                        cropped = self.model[:, xstart:xstop, ystart:ystop, zstart:zstop]
                        scaled = cropped * charge
                        self.field += scaled
        self.time += self.timestep




class Dipole:
    def __init__(self, efield, frequency):
        self.frequency = frequency
        self.efield = efield
        width = LIGHT_SPEED / frequency / 2 / efield.scale
        half = width / 2
        ctr = [dim / 2 for dim in efield.size[:2]]
        rang = [ctr[0] - half, ctr[0] + half]
        pxrang = [max(0, int(math.floor(rang[0]))), min(efield.size[0] -1, int(math.ceil(rang[1])))]
        self.values = numpy.linspace((pxrang[0] - ctr[0]) / width, (pxrang[1] - ctr[0]) / width, pxrang[1] - pxrang[0] + 1) * math.pi
        self.start = [pxrang[0], int(ctr[1])]
        
    def update(self):
        for i in range(len(self.values)):
            self.efield.chargefield[self.start[0] + i, self.start[1]] = math.sin(self.values[i]) * math.sin(2 * math.pi * self.efield.time * self.frequency)




class EFieldVisualizer:
    def __init__(self, efield):
        self.efield = efield
    
    def fieldRgb(self):
        plane = self.efield.field[0, ..., 0, :]
        col = plane / 10000000000000 + 0.5
        return col
        
    def magnitudes(self):
        plane = self.efield.field[0, ..., 0, :]
        mag = numpy.sqrt(numpy.einsum("xyp,xyp->xy", plane, plane))
        stitched = numpy.repeat(mag[..., None], 3, 2)
        col = stitched / 10000000000000
        return col

    def charges(self):
        plane = self.efield.chargefield
        R = plane * (plane > 0)
        G = numpy.zeros(plane.shape)
        B = numpy.absolute(plane * (plane < 0))
        stitched = numpy.concatenate([R, G, B], 2)
        return stitched

    def chargesAlpha(self):
        plane = self.efield.chargefield
        R = plane > 0
        G = numpy.zeros(plane.shape)
        B = plane < 0
        A = numpy.absolute(plane)
        stitched = numpy.concatenate([R, G, B, A], 2)
        return stitched




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




w = Window((800, 400))
f = EField((51, 51, 1), 0.01, 1 / 2400000000 / 16)
d = Dipole(f, 2400000000)
fv = EFieldVisualizer(f)

while w.update():
    w.fill([0, 0, 0])
    w.drawArray(fv.magnitudes(), [0.5, 1], [0, 0])
    w.drawArray(fv.chargesAlpha(), [0.5, 1], [0, 0])
    w.drawArray(fv.fieldRgb(), [0.5, 1], [0.5, 0])
    #d.update()
    f.chargefield[24, 24] = 1
    f.update()
    w.sleep(20)

w.close()