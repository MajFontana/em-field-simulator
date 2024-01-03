import numpy
import math
import cupy
import os




class FieldEngine:
    def __init__(self, field):
        self.field = field

        # Cupy data structures
        self.gaussmodel = cupy.array(self.field.gaussmodel, cupy.float32)
        #self.cudafaradaymodel = cupy.array(self.field.faradaymodel, cupy.float32)
        self.amperemodel = cupy.array(self.field.amperemodel, cupy.float32)

        # Setup variables
        self.BLOCK_SIZE = 512

        self.fieldsize = cupy.array((self.field.timesize, *self.field.size), cupy.int_)
        self.modelsize = cupy.array([dim * 2 - 1 for dim in self.field.size], cupy.int_)
        self.inputsperpoint = int(self.fieldsize[1] * self.fieldsize[2] * self.fieldsize[3])
        self.blocksperpoint = int(math.ceil(self.inputsperpoint / self.BLOCK_SIZE))

        self.blocksperfield = int(self.fieldsize[0] * self.fieldsize[1] * self.fieldsize[2] * self.fieldsize[3] * 3 * self.blocksperpoint)

        with open(os.path.join(os.path.dirname(__file__), "cudakernels.cu"), "r") as file:
            self.gaussLawKernel = cupy.RawModule(code=file.read()).get_function("gaussLawKernel")

    def compute(self):
        # Load from field
        chargedensity = cupy.array(self.field.chargedensity, cupy.float32)
        currentdensity = cupy.array(self.field.currentdensity, cupy.float32)

        efield = cupy.array(self.field.efield, cupy.float32)
        bfield = cupy.array(self.field.bfield, cupy.float32)

        # Compute new fields
        self.gaussLawKernel((self.blocksperfield,), (self.BLOCK_SIZE,), (efield, chargedensity, self.fieldsize, self.modelsize, self.inputsperpoint, self.blocksperpoint, self.gaussmodel))

        # Store to field
        self.field.efield = cupy.asnumpy(efield)
        self.field.bfield = cupy.asnumpy(bfield)