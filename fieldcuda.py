import numpy
import math
import cupy




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
        radiicubed = numpy.arange(0, math.hypot(math.hypot(abssize[0], abssize[1]), abssize[2]) + dstep, dstep) ** 3
        self.timesize = len(radiicubed) - 1
        tslicemask = numpy.array([(distcubed > radiicubed[i]) & (distcubed < radiicubed[i + 1]) for i in range(self.timesize)])[..., None]

        # Compute models for Maxwell's equations
        self.gaussmodel = divergence * (1 / VAC_PERMITTIVITY / 4 / math.pi) * tslicemask
        self.faradaymodel = 0 # TODO: curl due to change in magnetic field
        self.amperemodel = (curl * (VAC_PERMEABILITY / 4 / math.pi))[:, None, ...] * tslicemask

        self.cudagaussmodel = cupy.array(self.gaussmodel, cupy.float)
        #self.cudafaradaymodel = cupy.array(self.faradaymodel, cupy.double)
        self.cudaamperemodel = cupy.array(self.amperemodel, cupy.float)

        self.cudaefield = cupy.zeros((self.timesize, *size, 3), cupy.float)
        self.cudabfield = cupy.zeros((self.timesize, *size, 3), cupy.float)
        self.time = 0
        self.cudachargedensity = cupy.zeros(size, cupy.float)
        self.cudacurrentdensity = cupy.zeros((*size, 3), cupy.float)

        # Setup cuda
        self.BLOCK_SIZE = 512

        self.cudafieldsize = cupy.array((self.timesize, *self.size), cupy.int)
        self.cudamodelsize = cupy.array([dim * 2 - 1 for dim in self.size], cupy.int)
        self.cudapointsperinp = cupy.int(self.cudafieldsize[1] * self.cudafieldsize[2] * self.cudafieldsize[3])
        self.cudablocksperpoint = cupy.int(int(math.ceil(self.cudapointsperinp / self.BLOCK_SIZE)))
        self.blockspergrid = int(self.cudafieldsize[0] * self.cudafieldsize[1] * self.cudafieldsize[2] * self.cudafieldsize[3] * 3 * self.cudablocksperpoint)

        with open("fieldkernels.h", "r") as file:
            code = file.read()
        self.gaussLawKernel = cupy.RawModule(code=code).get_function("gaussLawKernel")

        self.updateHost()

    def update(self):
        self.cudachargedensity = cupy.array(self.chargedensity, cupy.float)
        self.cudacurrentdensity = cupy.array(self.currentdensity, cupy.float)

        # Advance time
        self.cupyefield = cupy.roll(self.cudaefield, -1, 0)
        self.cudaefield[-1] = 0

        self.cudabfield = cupy.roll(self.cudabfield, -1, 0)
        self.cudabfield[-1] = 0

        self.time += self.timestep

        # Compute new fields
        self.gaussLawKernel((self.blockspergrid,), (self.BLOCK_SIZE,), (self.cudaefield, self.cudachargedensity, self.cudafieldsize, self.cudamodelsize, self.cudapointsperinp, self.cudablocksperpoint, self.cudagaussmodel))

        self.updateHost()

    def updateHost(self):
        self.efield = cupy.asnumpy(self.cudaefield)
        self.bfield = cupy.asnumpy(self.cudabfield)
        self.chargedensity = cupy.asnumpy(self.cudachargedensity)
        self.currentdensity = cupy.asnumpy(self.cudacurrentdensity)




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