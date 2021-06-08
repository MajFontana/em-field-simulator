import numpy
import math




VAC_PERMITTIVITY = 8.8541878128 * 10**(-12)
VAC_PERMEABILITY = 1.25663706212 * 10**(-6)
LIGHT_SPEED = 299792458




class Field:
    def __init__(self, size, scale, timestep, compute="numpy"):
        self.size = tuple(size)
        self.scale = scale
        abssize = [(dim - 1) * scale for dim in size]

        # Compute model coordinates
        modelsize = [dim * 2 - 1 for dim in size]
        axes = [numpy.linspace(-abssize[i], abssize[i], modelsize[i]) if modelsize[i] > 1 else numpy.array([0]) for i in range(3)]
        coords = numpy.stack(numpy.meshgrid(*axes, indexing="ij"), axis=-1)
        self.modelcenter = tuple([dim - 1 for dim in self.size])

        # Compute divergence and curl models
        distcubed = numpy.einsum("xyzp,xyzp->xyz", coords, coords) ** 1.5
        with numpy.errstate(divide="ignore", invalid="ignore"):
            divergence = coords / distcubed[..., None]
        divergence[self.modelcenter] = 0
        curl = numpy.array([numpy.cross(basisvec, divergence) for basisvec in numpy.eye(3)])

        self.timestep = timestep
        dstep = self.timestep * LIGHT_SPEED
        
        # Compute spacetime propagation mask
        radiicubed = numpy.arange(0, math.hypot(math.hypot(abssize[0], abssize[1]), abssize[2]) + dstep, dstep) ** 3
        self.timesize = len(radiicubed) - 1
        tslicemask = numpy.array([(distcubed > radiicubed[i]) & (distcubed < radiicubed[i + 1]) for i in range(self.timesize)])[..., None]

        # Compute models for Maxwell's equations
        self.gaussmodel = divergence * (1 / VAC_PERMITTIVITY / 4 / math.pi) * tslicemask
        #self.faradaymodel = 0
        self.amperemodel = (curl * (VAC_PERMEABILITY / 4 / math.pi))[:, None, ...] * tslicemask

        # Working arrays
        self.efield = numpy.zeros((self.timesize, *size, 3))
        self.bfield = numpy.zeros((self.timesize, *size, 3))
        self.chargedensity = numpy.zeros(size)
        self.currentdensity = numpy.zeros((*size, 3))
        self.time = 0

        if compute == "numpy":
            from .fieldcompute import numpycompute
            self.engine = numpycompute.FieldEngine(self)
        elif compute == "cuda":
            from .fieldcompute import cudacompute
            self.engine = cudacompute.FieldEngine(self)
        else:
            self.engine = None

    def update(self):
        # Advance time
        self.efield = numpy.roll(self.efield, -1, 0)
        self.efield[-1] = 0

        self.bfield = numpy.roll(self.bfield, -1, 0)
        self.bfield[-1] = 0

        self.time += self.timestep
        
        # Compute new fields
        if self.engine != None:
            self.engine.compute()

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