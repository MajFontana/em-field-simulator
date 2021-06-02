import numba.cuda
import math
import numpy




BLOCK_SIZE = 256




def setup(size, gauss):
    global block_size
    global fieldsize
    global gaussmodel
    global pointsperinp
    global blocksperpoint

    block_size = numpy.uint32(BLOCK_SIZE)
    fieldsize = numpy.array(size, numpy.uint32)
    gaussmodel = gauss.astype(numpy.float32)
    pointsperinp = numpy.uint32(fieldsize[1] * fieldsize[2] * fieldsize[3])
    blocksperpoint = numpy.uint32(int(math.ceil(pointsperinp / block_size)))

    print("Field size:", fieldsize)
    print("Blocks per point:", blocksperpoint)
    print("Points per input (threads per block group):", pointsperinp)
    print("Points per field (block group count):", fieldsize[0] * fieldsize[1] * fieldsize[2] * fieldsize[3] * 3)
    print("Total block count:", blocksperpoint * fieldsize[0] * fieldsize[1] * fieldsize[2] * fieldsize[3] * 3)
    print("Total threads:",  blocksperpoint * fieldsize[0] * fieldsize[1] * fieldsize[2] * fieldsize[3] * 3 * block_size)
    print("Used threads:", fieldsize[0] * fieldsize[1] * fieldsize[2] * fieldsize[3] * 3 * pointsperinp)
    print("Cuda available:", numba.cuda.is_available())

@numba.cuda.jit
def gaussLawKernel(efield, chargedensity):
    inpidx = (numba.cuda.blockIdx.x % blocksperpoint) * block_size + numba.cuda.threadIdx.x
    if inpidx < pointsperinp:
        fieldidx = numba.cuda.blockIdx.x // blocksperpoint

        cudagaussmodel = numba.cuda.const.array_like(gaussmodel)

        d = fieldidx % numpy.uint32(3)
        fieldz = fieldidx // numpy.uint32(3)  % fieldsize[3]
        fieldy = fieldidx // numpy.uint32(3) // fieldsize[3] % fieldsize[2]
        fieldx = fieldidx // numpy.uint32(3) // fieldsize[3] // fieldsize[2] % fieldsize[1]
        t = fieldidx // numpy.uint32(3) // fieldsize[3] // fieldsize[2] // fieldsize[1]

def compute(efield, chargedensity):
    gaussLawKernel[blocksperpoint * fieldsize[0] * fieldsize[1] * fieldsize[2] * fieldsize[3] * 3, block_size](efield.astype(numpy.float32), chargedensity.astype(numpy.float32))