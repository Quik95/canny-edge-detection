__kernel void grayscale_image(__constant float* inputImage, __global float* outputImage) {
    int i = get_global_id(0);
    float red = inputImage[i * 3];
    float green = inputImage[i * 3 + 1];
    float blue = inputImage[i * 3 + 2];
    float sum = 0.299f * red + 0.587f * green + 0.114f * blue;
    outputImage[i] = sum;
}