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


extern "C" __constant__ int TABLE_LEN;
extern "C" __constant__ unsigned char TABLE[500];  // x,y,opacity

extern "C"
__global__ void RGBToYKernel(unsigned char *src, unsigned char *dst, int width, int height, int pitch){
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if ((gidx < width) && (gidy < height)) {
        int pos = gidx * 3 + gidy * pitch;
        // Y = 0.299 R + 0.587 G + 0.114 B
        // uchar3 value = src[pos];
        /*float Y = 0.299f * value.x + 0.587f * value.y + 0.114f * value.z;
        unsigned char y = (unsigned char)min(255, (int)Y);
        uchar3 pixel;
        pixel.x = 1;
        pixel.y = 2;
        pixel.z = 3;*/
        dst[pos] = src[pos];
        dst[pos+1] = src[pos+1];
        dst[pos+2] = src[pos+2] / 2;
    }
}

extern "C"
__global__ void BlurKernel(unsigned char *src, unsigned char *dst, int width, int height, int pitch){
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    int pos = gidx * 3 + gidy * pitch;

    unsigned long totalOpacity = 50;
    unsigned long R = 0;
    unsigned int G = 0;
    unsigned int B = 0;
    int index;
    for (index = 0; index < 2; index++) {
        int tableX = gidx + (signed char)(TABLE[index]);
        int tableY = gidy + (signed char)(TABLE[index + 1]);
        unsigned char tableOpacity = 255; // TABLE[index + 2];
        if (tableX >= 0 && tableX < width && tableY >= 0 && tableY < height) {
            int p = tableX * 3 + tableY * pitch;
            totalOpacity += tableOpacity;
            R += src[pos];
            G += src[p + 1];
            B += src[p + 2];
        }

//        dst[pos] = (unsigned char)R; //(R / (1.0 * totalOpacity / 255));
//        if (tableX >= 0 && tableX < width && tableY >= 0 && tableY < height) {
//            dst[pos+1] = (unsigned char)200;
//        }
    }

    dst[pos] = (unsigned char) (R / (1.0 * totalOpacity / 255));
//    dst[pos+1] = src[pos+1];
//    dst[pos] = (unsigned char)R;
//    dst[pos+1] = (unsigned char)gidx;
//    dst[pos+2] = (unsigned char)gidy;
}



extern "C"
__global__ void Blur2Kernel(unsigned char *src, unsigned char *dst, int width, int height, int xOffset, int yOffset, double opacity){
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int writeX = gidx % width;
    int writeY = gidx / width;
    int readX = writeX - xOffset;
    int readY = writeY - yOffset;

    if (readX >= 0 && readX <= width && readY >= 0 && readY <= height) {
        int readPos = readX * 3 + readY * width * 3;
        int writePos = writeX * 3 + writeY * width * 3;

        int r = src[readPos];
        int g = src[readPos + 1];
        int b = src[readPos + 2];

        dst[writePos] += r * opacity;
        dst[writePos + 1] += g * opacity;
        dst[writePos + 2] += b * opacity;
    }
}


