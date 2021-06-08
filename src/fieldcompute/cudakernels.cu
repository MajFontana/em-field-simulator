extern "C" __global__
void gaussLawKernel(float* efield, float* chargedensity, unsigned int* fieldsize, unsigned int* modelsize, unsigned int inputsperpoint, unsigned int blocksperpoint, float* gaussmodel) {
    // Allocate and initalize shared memory for parallel reduction
    extern __shared__ float blockdata[512];
    blockdata[threadIdx.x] = 0.0;
    __syncthreads();

    // Compute indices
    unsigned int fieldidx = blockIdx.x / blocksperpoint;
    int inpidx = (blockIdx.x % blocksperpoint) * blockDim.x + threadIdx.x;

    if (inpidx < inputsperpoint) {
        // Compute coordinates
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

        // Compute model index
        unsigned int modelidx = t * 3 * modelsize[2] * modelsize[1] * modelsize[0] + modelx * 3 * modelsize[2] * modelsize[1] + modely * 3 * modelsize[2] + modelz * 3 + d;

        // Apply model to field
        blockdata[threadIdx.x] = gaussmodel[modelidx] * chargedensity[inpidx];
    }
    __syncthreads();

    // Parallel reduction - sum effects of all points in block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            blockdata[threadIdx.x] += blockdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Sum effects of entire block group
    if (threadIdx.x == 0) {
        atomicAdd(&efield[fieldidx], blockdata[0]);
    }
}