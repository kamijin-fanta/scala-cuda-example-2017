#pragma warning(disable:4819)

extern "C"
__global__ void add(int n, float *a, float *b, float *sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        sum[i] = a[i] + b[i];
    }

}

extern "C"
__global__ void RGBToYKernel(unsigned char *src, unsigned char *dst, int width, int height){
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int x = gidx % width;
    int y = gidx / width;

        // Y = 0.299 R + 0.587 G + 0.114 B

    if (x >= 0 && x <= width && y >= 0 && y <= height) {
        int pos = x * 3 + y * width * 3;

        int r = src[pos];
        int g = src[pos + 1];
        int b = src[pos + 2];
        int y = 0.299 * r + 0.587 * g + 0.114 * b;

        dst[pos] += y;
        dst[pos + 1] += y;
        dst[pos + 2] += y;
    }
}


