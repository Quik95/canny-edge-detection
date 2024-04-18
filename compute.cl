__kernel void grayscale_image(
    __global float* inputImage,
    __global float* outputImage
){
    int colIndex = get_global_id(0);
    int rowIndex = get_global_id(1);
    int imageWidth = get_global_size(0);
    int index = rowIndex * imageWidth + colIndex;
    
    float red = inputImage[index * 3 + 0];
    float green = inputImage[index * 3 + 1];
    float blue = inputImage[index * 3 + 2];
    float sum = 0.299f * red + 0.587f * green + 0.114f * blue;
    outputImage[index] = sum;
}

__kernel void gaussian_blur(
    __global float* inputImage,
    __global float* outputImage
){
    int colIndex = get_global_id(0);
    int rowIndex = get_global_id(1);
    int imageWidth = get_global_size(0);
    
    float newPixelValue = 0.0f;

    
    // Gaussian blur static table
    const float gaussianTable[25] = {
            1/273.0f, 4/273.0f, 7/273.0f, 4/273.0f, 1/273.0f,
            4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f,
            7/273.0f, 26/273.0f, 41/273.0f, 26/273.0f, 7/273.0f,
            4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f,
            1/273.0f, 4/273.0f, 7/273.0f, 4/273.0f, 1/273.0f
    };
    
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            int neighborCol = colIndex + j;
            int neighborRow = rowIndex + i;
            
            if (neighborCol >= 0 && neighborCol < imageWidth && neighborRow >= 0) {
                int neighborIndex = neighborRow * imageWidth + neighborCol;
                newPixelValue += inputImage[neighborIndex] * gaussianTable[(i + 2) * 3 + (j + 2)];
            }
        }
    }
    
    outputImage[rowIndex * imageWidth + colIndex] = clamp(newPixelValue, 0.0f, 1.0f);
}