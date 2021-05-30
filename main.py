import numpy
import pygame
import math




COULOMB_CONST = 8987551792.3 # Might not need this anymore
VAC_PERMITTIVITY = 8.8541878128 * 10**(-12)
VAC_PERMEABILITY = 1.25663706212 * 10**(-6)
LIGHT_SPEED = 299792458




class Field:
    def __init__(self, size, scale, timestep):
        self.size = size
        self.scale = scale
        abssize = [(dim - 1) * scale for dim in size]

        modelsize = [dim * 2 - 1 for dim in size]
        axes = [numpy.linspace(-abssize[i], abssize[i], modelsize[i]) if modelsize[i] > 1 else numpy.array([0]) for i in range(3)]
        coords = numpy.stack(numpy.meshgrid(*axes, indexing="ij"), axis=-1)
        self.modelcenter = tuple([int(dim / 2) for dim in modelsize])

        distcubed = numpy.einsum("xyzp,xyzp->xyz", coords, coords) ** 1.5
        with numpy.errstate(divide="ignore", invalid="ignore"):
            divergence = coords / distcubed[..., None]
        divergence[self.modelcenter] = 0
        curl = numpy.array([numpy.cross(basisvec, divergence) for basisvec in numpy.eye(3)])

        self.timestep = timestep
        dstep = timestep * LIGHT_SPEED
        
        radiicubed = list(numpy.arange(0, math.hypot(math.hypot(abssize[0], abssize[1]), abssize[2]) + dstep, dstep) ** 3)
        tsize = len(radiicubed)
        print(radiicubed)
        tslicemask = numpy.array([(distcubed > radiicubed[i]) & (distcubed < radiicubed[i + 1]) for i in range(tsize - 1)])[..., None]

        self.efielddivmodel = divergence * (1 / VAC_PERMITTIVITY / 4 / math.pi) * tslicemask
        self.efieldcurlmodel = 0 # TODO: curld due to change in magnetic field
        self.bfieldcurlmodel = (curl * (VAC_PERMEABILITY / 4 / math.pi))[:, None, ...] * tslicemask

        self.efield = numpy.zeros((tsize - 1, *size, 3))
        self.bfield = numpy.zeros((tsize - 1, *size, 3))
        self.chargefield = numpy.zeros(size)
        self.currentdensity = numpy.zeros((*size, 3))
        self.time = 0

        import matplotlib.pyplot as plt

        ax = plt.figure().add_subplot(projection='3d')

        # Make the grid
        x, y, z = numpy.meshgrid(numpy.linspace(-5, 5, modelsize[0]),
                              numpy.linspace(-5, 5, modelsize[1]),
                              numpy.linspace(-5, 5, modelsize[2]))

        # Make the direction data for the arrows
        u = self.bfieldcurlmodel[0][0][..., 0]
        v = self.bfieldcurlmodel[0][0][..., 1]
        w = self.bfieldcurlmodel[0][0][..., 2]

        ax.quiver(x, y, z, v, u, w, length=1000, normalize=False)

        plt.show()

    def update(self):
        self.efield = numpy.roll(self.efield, -1, 0)
        self.efield[-1] = 0
        self.bfield = numpy.roll(self.bfield, -1, 0)
        self.bfield[-1] = 0
        for x in range(self.size[0]):
            xstart = self.modelcenter[0] - x
            xstop = xstart + self.size[0]
            for y in range(self.size[1]):
                ystart = self.modelcenter[1] - y
                ystop = ystart + self.size[1]
                for z in range(self.size[2]):
                    zstart = self.modelcenter[2] - z
                    zstop = zstart + self.size[2]
                    
                    charge = self.chargefield[x, y, z]
                    if charge != 0:
                        croppedE = self.efielddivmodel[:, xstart:xstop, ystart:ystop, zstart:zstop]
                        scaledE = croppedE * charge
                        self.efield += scaledE

                    current = self.currentdensity[x, y, z]
                    if any(current):
                        croppedB = self.bfieldcurlmodel[..., xstart:xstop, ystart:ystop, zstart:zstop, :]
                        scaledB = croppedB * current # TODO: Need to verify that this does the right thing - multiply the model with each axis of the current vector
                        self.bfield += numpy.sum(scaledB, 0)
        self.time += self.timestep




# TODO: Need to redo this to simulate both charge and current
class Dipole:
    def __init__(self, efield, frequency):
        self.frequency = frequency
        self.efield = efield
        width = LIGHT_SPEED / frequency / 2 / efield.scale
        half = width / 2
        ctr = [dim / 2 for dim in efield.size]
        rang = [ctr[0] - half, ctr[0] + half]
        pxrang = [max(0, int(math.floor(rang[0]))), min(efield.size[0] -1, int(math.ceil(rang[1])))]
        self.values = numpy.linspace((pxrang[0] - ctr[0]) / width, (pxrang[1] - ctr[0]) / width, pxrang[1] - pxrang[0] + 1) * math.pi
        self.start = [pxrang[0], int(ctr[1]), int(ctr[2])]
        
    def update(self):
        for i in range(len(self.values)):
            self.efield.chargefield[self.start[0] + i, self.start[1], self.start[2]] = math.sin(self.values[i]) * math.sin(2 * math.pi * self.efield.time * self.frequency)




class EFieldVisualizer:
    def __init__(self, efield, z):
        self.efield = efield
        self.z = z
    
    def fieldRgb(self):
        plane = self.efield.efield[0, ..., self.z, :]
        col = plane / 10000000000000 + 0.5
        return col
        
    def magnitudes(self):
        plane = self.efield.efield[0, ..., self.z, :]
        mag = numpy.sqrt(numpy.einsum("xyp,xyp->xy", plane, plane))
        stitched = numpy.repeat(mag[..., None], 3, 2)
        col = stitched / 10000000000000
        return col

    def charges(self):
        plane = self.efield.chargefield[..., self.z:self.z+1]
        R = plane * (plane > 0)
        G = numpy.zeros(plane.shape)
        B = numpy.absolute(plane * (plane < 0))
        stitched = numpy.concatenate([R, G, B], 2)
        return stitched

    def chargesAlpha(self):
        plane = self.efield.chargefield[..., self.z:self.z+1]
        R = plane > 0
        G = numpy.zeros(plane.shape)
        B = plane < 0
        A = numpy.absolute(plane)
        stitched = numpy.concatenate([R, G, B, A], 2)
        return stitched

    def currentAlpha(self):
        plane = self.efield.currentdensity[..., self.z, :]
        col = plane + 0.5
        mag = numpy.sqrt(numpy.einsum("xyp,xyp->xy", plane, plane))
        stitched = numpy.append(col, mag[..., None], -1)
        return stitched

    def bFieldRgb(self):
        plane = self.efield.bfield[0, ..., self.z, :]
        col = plane * 1000 + 0.5
        return col

    def bMagnitudes(self):
        plane = self.efield.bfield[0, ..., self.z, :]
        mag = numpy.sqrt(numpy.einsum("xyp,xyp->xy", plane, plane))
        stitched = numpy.repeat(mag[..., None], 3, 2)
        col = stitched * 1000
        return col




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




w = Window((800, 800))
f = Field((11, 11, 11), 0.01, 0.01)
#d = Dipole(f, 2400000000)
f.currentdensity[5, 5, :] = (0, 0, 1)
fv = EFieldVisualizer(f, 5)

while w.update():
    w.fill([0, 0, 0])
    w.drawArray(fv.magnitudes(), [0.5, 0.5], [0, 0])
    w.drawArray(fv.chargesAlpha(), [0.5, 0.5], [0, 0])
    w.drawArray(fv.fieldRgb(), [0.5, 0.5], [0.5, 0])
    w.drawArray(fv.bMagnitudes(), [0.5, 0.5], [0, 0.5])
    w.drawArray(fv.currentAlpha(), [0.5, 0.5], [0, 0.5])
    w.drawArray(fv.bFieldRgb(), [0.5, 0.5], [0.5, 0.5])
    #d.update()
    f.update()
    w.sleep(20)

w.close()
