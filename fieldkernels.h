extern "C" __global__
void gaussLawKernel(float* efield, float* chargedensity, unsigned int* fieldsize, unsigned int* modelsize, unsigned int pointsperinp, unsigned int blocksperpoint, float* gaussmodel) {
    extern __shared__ float blockdata[];

    int inpidx = (blockIdx.x % blocksperpoint) * blockDim.x + threadIdx.x;
    blockdata[threadIdx.x] = 0.0;
    __syncthreads();

    if (inpidx < pointsperinp) {
        unsigned int fieldidx = blockIdx.x / blocksperpoint;

        unsigned int d = fieldidx % 3;
        unsigned int fieldz = fieldidx / 3 % fieldsize[3];
        unsigned int fieldy = fieldidx / 3 / fieldsize[3] % fieldsize[2];
        unsigned int fieldx = fieldidx / 3 / fieldsize[3] / fieldsize[2] % fieldsize[1];
        unsigned int t = fieldidx / 3 / fieldsize[3] / fieldsize[2] / fieldsize[1];

        unsigned int inpz = inpidx % fieldsize[3];
        unsigned int inpy = inpidx / fieldsize[3] % fieldsize[2];
        unsigned int inpx = inpidx / fieldsize[3] / fieldsize[2];

        unsigned int modeloffz = fieldsize[3] - fieldz - 1;
        unsigned int modeloffy = fieldsize[2] - fieldy - 1;
        unsigned int modeloffx = fieldsize[1] - fieldx - 1;

        unsigned int modelz = modeloffz + inpz;
        unsigned int modely = modeloffy + inpy;
        unsigned int modelx = modeloffx + inpx;

        unsigned int modelidx = t * 3 * modelsize[2] * modelsize[1] * modelsize[0] + modelx * 3 * modelsize[2] * modelsize[1] + modely * 3 * modelsize[2] + modelz * 3 + d;

        blockdata[threadIdx.x] = gaussmodel[modelidx] * chargedensity[inpidx];
        __syncthreads();

        for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
            int index = 2 * stride * threadIdx.x;
            if (index < blockDim.x) {
                blockdata[index] += blockdata[index + stride];
            }
            __syncthreads();
        }
        
        if (threadIdx.x == 0) {
            atomicAdd(&efield[fieldidx], blockdata[0]);
        }
    }
}