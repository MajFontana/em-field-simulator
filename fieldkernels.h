extern "C" __global__
void gaussLawKernel(float* efield, float* chargedensity, int* fieldsize, int* modelsize, int pointsperinp, int blocksperpoint, float* gaussmodel) {
    int inpidx = (blockIdx.x % blocksperpoint) * blockDim.x + threadIdx.x;

    if (inpidx < pointsperinp) {
        int fieldidx = blockIdx.x / blocksperpoint;

        int d = fieldidx % 3;
        int fieldz = fieldidx / 3 % fieldsize[3];
        int fieldy = fieldidx / 3 / fieldsize[3] % fieldsize[2];
        int fieldx = fieldidx / 3 / fieldsize[3] / fieldsize[2] % fieldsize[1];
        int t = fieldidx / 3 / fieldsize[3] / fieldsize[2] / fieldsize[1];

        int inpz = inpidx % fieldsize[3];
        int inpy = inpidx / fieldsize[3] % fieldsize[2];
        int inpx = inpidx / fieldsize[3] / fieldsize[2];

        int modeloffz = fieldsize[3] - fieldz - 1;
        int modeloffy = fieldsize[2] - fieldy - 1;
        int modeloffx = fieldsize[1] - fieldx - 1;

        int modelz = modeloffz + inpz;
        int modely = modeloffy + inpy;
        int modelx = modeloffx + inpx;

        int modelidx = t * 3 * modelsize[2] * modelsize[1] * modelsize[0] + modelx * 3 * modelsize[2] * modelsize[1] + modely * 3 * modelsize[2] + modelz * 3 + d;

        atomicAdd(&efield[fieldidx], gaussmodel[modelidx] * chargedensity[inpidx]);
    }
}