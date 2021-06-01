import numpy
import math




VAC_PERMITTIVITY = 8.8541878128 * 10**(-12)
VAC_PERMEABILITY = 1.25663706212 * 10**(-6)
LIGHT_SPEED = 299792458




class Field:
    def __init__(self, size, scale, timestep):
        self.size = size
        self.scale = scale
        abssize = [(dim - 1) * scale for dim in size]

        # Compute model coordinates
        modelsize = [dim * 2 - 1 for dim in size]
        axes = [numpy.linspace(-abssize[i], abssize[i], modelsize[i]) if modelsize[i] > 1 else numpy.array([0]) for i in range(3)]
        coords = numpy.stack(numpy.meshgrid(*axes, indexing="ij"), axis=-1)
        self.modelcenter = tuple([int(dim / 2) for dim in modelsize])

        # Compute divergence and curl models
        distcubed = numpy.einsum("xyzp,xyzp->xyz", coords, coords) ** 1.5
        with numpy.errstate(divide="ignore", invalid="ignore"):
            divergence = coords / distcubed[..., None]
        divergence[self.modelcenter] = 0
        curl = numpy.array([numpy.cross(basisvec, divergence) for basisvec in numpy.eye(3)])

        self.timestep = timestep
        dstep = self.timestep * LIGHT_SPEED
        
        # Compute spacetime propagation mask
        radiicubed = list(numpy.arange(0, math.hypot(math.hypot(abssize[0], abssize[1]), abssize[2]) + dstep, dstep) ** 3)
        tsize = len(radiicubed)
        tslicemask = numpy.array([(distcubed > radiicubed[i]) & (distcubed < radiicubed[i + 1]) for i in range(tsize - 1)])[..., None]

        # Compute models for Maxwell's equations
        self.gaussmodel = divergence * (1 / VAC_PERMITTIVITY / 4 / math.pi) * tslicemask
        self.faradaymodel = 0 # TODO: curl due to change in magnetic field
        self.amperemodel = (curl * (VAC_PERMEABILITY / 4 / math.pi))[:, None, ...] * tslicemask

        self.efield = numpy.zeros((tsize - 1, *size, 3))
        self.bfield = numpy.zeros((tsize - 1, *size, 3))
        self.time = 0
        self.chargedensity = numpy.zeros(size)
        self.currentdensity = numpy.zeros((*size, 3))

    def update(self):
        # Advance time
        self.efield = numpy.roll(self.efield, -1, 0)
        self.efield[-1] = 0

        self.bfield = numpy.roll(self.bfield, -1, 0)
        self.bfield[-1] = 0

        self.time += self.timestep
        
        # Compute new fields
        for x in range(self.size[0]):
            xstart = self.modelcenter[0] - x
            xstop = xstart + self.size[0]
            for y in range(self.size[1]):
                ystart = self.modelcenter[1] - y
                ystop = ystart + self.size[1]
                for z in range(self.size[2]):
                    zstart = self.modelcenter[2] - z
                    zstop = zstart + self.size[2]
                    
                    # E field due to charge density
                    charge = self.chargedensity[x, y, z]
                    if charge != 0:
                        croppedE = self.gaussmodel[:, xstart:xstop, ystart:ystop, zstart:zstop]
                        scaledE = croppedE * charge
                        self.efield += scaledE

                    # B field due to current density
                    current = self.currentdensity[x, y, z]
                    if any(current):
                        croppedB = self.amperemodel[:, :, xstart:xstop, ystart:ystop, zstart:zstop]
                        scaledB = croppedB * current[:, None, None, None, None, None]
                        self.bfield += numpy.sum(scaledB, 0)




class Dipole:
    def __init__(self, field, frequency):
        self.frequency = frequency
        self.field = field
        width = LIGHT_SPEED / frequency / 2 / field.scale
        half = width / 2
        ctr = [dim / 2 for dim in field.size]
        rang = [ctr[0] - half, ctr[0] + half]
        pxrang = [max(0, int(math.floor(rang[0]))), min(field.size[0] - 1, int(math.ceil(rang[1])))]
        self.values = numpy.linspace((pxrang[0] - ctr[0]) / width, (pxrang[1] - ctr[0]) / width, pxrang[1] - pxrang[0] + 1) * math.pi
        self.start = [pxrang[0], int(ctr[1]), int(ctr[2])]
        
    def update(self):
        for i in range(len(self.values)):
            self.field.chargedensity[self.start[0] + i, self.start[1], self.start[2]] = math.sin(self.values[i]) * math.sin(2 * math.pi * self.field.time * self.frequency)
            self.field.currentdensity[self.start[0] + i, self.start[1], self.start[2]] = [math.cos(self.values[i]) * math.cos(2 * math.pi * self.field.time * self.frequency), 0, 0]