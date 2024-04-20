constant sampler_t
sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_MIRRORED_REPEAT | CLK_FILTER_NEAREST;

__kernel void grayscale_image(
        read_only image2d_t inputImage,
        write_only image2d_t outputImage
){
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float4 pixel = read_imagef(inputImage, sampler, coord);
    float sum = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;

    write_imagef(outputImage, coord, (float4)(sum, sum, sum, 1.0f));
}

__kernel void gaussian_blur(
    __global float* inputImage,
    __global float* outputImage
){
    int colIndex = get_global_id(0);
    int rowIndex = get_global_id(1);
    int imageWidth = get_global_size(0);
    
    float newPixelValue = 0.0f;

    
    const float gaussianTable[25] = {
            1/273.0f, 4/273.0f, 7/273.0f, 4/273.0f, 1/273.0f,
            4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f,
            7/273.0f, 26/273.0f, 41/273.0f, 26/273.0f, 7/273.0f,
            4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f,
            1/273.0f, 4/273.0f, 7/273.0f, 4/273.0f, 1/273.0f
    };

    int index = rowIndex * imageWidth + colIndex;
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            int neighborCol = colIndex + j;
            int neighborRow = rowIndex + i;
            
            if (neighborCol >= 0 && neighborCol < imageWidth && neighborRow >= 0) {
                int neighborIndex = neighborRow * imageWidth + neighborCol;
                newPixelValue += inputImage[neighborIndex] * gaussianTable[(i + 2) * 3 + (j + 2)];
            } else {
                newPixelValue += inputImage[index] * gaussianTable[(i + 2) * 3 + (j + 2)];
            }
        }
    }

    outputImage[index] = clamp(newPixelValue, 0.0f, 1.0f);
}

__kernel void sobel_filter(
    __global float* inputImage,
    __global float* outputImage
) {
    int colIndex = get_global_id(0);
    int rowIndex = get_global_id(1);
    int imageWidth = get_global_size(0);
    int imageHeight = get_global_size(1);

    const float sobelKernelX[9] = {
        -1.0f, 0.0f, 1.0f,
        -2.0f, 0.0f, 2.0f,
        -1.0f, 0.0f, 1.0f
    };
    const float sobelKernelY[9] = {
        -1.0f, -2.0f, -1.0f,
        0.0f, 0.0f, 0.0f,
        1.0f, 2.0f, 1.0f
    };

    float sobelX = 0.0f;
    float sobelY = 0.0f;

    int index = rowIndex * imageWidth + colIndex;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int neighborCol = colIndex + j;
            int neighborRow = rowIndex + i;

            if (neighborCol >= 0 && neighborCol < imageWidth && neighborRow >= 0) {
                int neighborIndex = neighborRow * imageWidth + neighborCol;
                sobelX += inputImage[neighborIndex] * sobelKernelX[(i + 1) * 3 + (j + 1)];
                sobelY += inputImage[neighborIndex] * sobelKernelY[(i + 1) * 3 + (j + 1)];
            } else {
                sobelX += inputImage[index] * sobelKernelX[(i + 1) * 3 + (j + 1)];
                sobelY += inputImage[index] * sobelKernelY[(i + 1) * 3 + (j + 1)];
            }
        }
    }

    float sobelMagnitude = sqrt(sobelX * sobelX + sobelY * sobelY);
    float orientation = atan2(sobelY, sobelX);
    orientation = orientation * (180.0f / M_PI);
    float orientationRounded = round(orientation / 45.0f) * 45.0f;
    orientation = fmod(orientationRounded + 180.0f, 180.0f);

    outputImage[index] = clamp(sobelMagnitude, 0.0f, 1.0f);
    outputImage[imageWidth * imageHeight + index] = clamp(orientation, 0.0f, 180.0f);
}

__kernel void edge_thinning(
    __global float* inputImage,
    __global float* outputImage
) {
    int colIndex = get_global_id(0);
    int rowIndex = get_global_id(1);
    int imageWidth = get_global_size(0);
    int imageHeight = get_global_size(1);
    int maxIndex = imageWidth * imageHeight;

    float q = 255.0f;
    float r = 255.0f;

    int angleIndex = rowIndex * imageWidth + colIndex + imageWidth * imageHeight;
    float angle = inputImage[angleIndex];

    if (angle >= 0 && angle < 22.5) {
        int qIndex = rowIndex * imageWidth + colIndex + 1;
        int rIndex = rowIndex * imageWidth + colIndex - 1;
        if (qIndex >= 0 && qIndex < maxIndex && rIndex >= 0 && rIndex < maxIndex) {
            q = inputImage[qIndex];
            r = inputImage[rIndex];
        }
    } else  if(angle >= 22.5 && angle < 67.5) {
        int qIndex = (rowIndex - 1) * imageWidth + colIndex - 1;
        int rIndex = (rowIndex + 1) * imageWidth + colIndex + 1;
        if (qIndex >= 0 && qIndex < maxIndex && rIndex >= 0 && rIndex < maxIndex) {
            q = inputImage[qIndex];
            r = inputImage[rIndex];
        }
    } else if(angle >= 67.5 && angle < 112.5) {
        int qIndex = (rowIndex - 1) * imageWidth + colIndex;
        int rIndex = (rowIndex + 1) * imageWidth + colIndex;
        if (qIndex >= 0 && qIndex < maxIndex && rIndex >= 0 && rIndex < maxIndex) {
            q = inputImage[qIndex];
            r = inputImage[rIndex];
        }
    } else if(angle >= 112.5 && angle < 157.5) {
        int qIndex = (rowIndex - 1) * imageWidth + colIndex + 1;
        int rIndex = (rowIndex + 1) * imageWidth + colIndex - 1;
        if (qIndex >= 0 && qIndex < maxIndex && rIndex >= 0 && rIndex < maxIndex) {
            q = inputImage[qIndex];
            r = inputImage[rIndex];
        }
    } else if(angle >= 157.5 && angle <= 180.0) {
        int qIndex = rowIndex * imageWidth + colIndex - 1;
        int rIndex = rowIndex * imageWidth + colIndex + 1;
        if (qIndex >= 0 && qIndex < maxIndex && rIndex >= 0 && rIndex < maxIndex) {
            q = inputImage[qIndex];
            r = inputImage[rIndex];
        }
    }

    int index = rowIndex * imageWidth + colIndex;
    float intensity = inputImage[index];
    if (intensity >= q && intensity >= r) {
        outputImage[index] = intensity;
    } else {
        outputImage[index] = 0.0f;
    }
}

#define STRONG_EDGE_VALUE 1.0f
#define WEAK_EDGE_VALUE 0.33f

#define STRONG_EDGE_THRESHOLD 0.12f
#define WEAK_EDGE_THRESHOLD 0.003f

__kernel void double_thresholding(
    __global float* inputImage,
    __global float* outputImage
) {
    int colIndex = get_global_id(0);
    int rowIndex = get_global_id(1);
    int imageWidth = get_global_size(0);
    int index = rowIndex * imageWidth + colIndex;

    // Find globally max intensity

    float intensity = inputImage[index];
    if (intensity >= STRONG_EDGE_THRESHOLD) {
        outputImage[index] = STRONG_EDGE_VALUE;
    } else if (intensity >= WEAK_EDGE_THRESHOLD) {
        outputImage[index] = WEAK_EDGE_VALUE;
    } else {
        outputImage[index] = 0.0f;
    }
}

#define CHECK_INDEX(index, imageWidth, imageHeight, fallback) ((index >= 0 && index < imageWidth * imageHeight) ? index : fallback)

__kernel void edge_histeresis(
        __global float *inputImage,
        __global float *outputImage
) {
    int colIndex = get_global_id(0);
    int rowIndex = get_global_id(1);
    int imageWidth = get_global_size(0);
    int imageHeight = get_global_size(1);
    int index = rowIndex * imageWidth + colIndex;

    if (inputImage[index] == STRONG_EDGE_VALUE) {
        outputImage[index] = STRONG_EDGE_VALUE;
        return;
    }

    if (inputImage[index] == WEAK_EDGE_VALUE && (
            inputImage[CHECK_INDEX(colIndex + 1 + rowIndex * imageWidth, imageWidth, imageHeight, index)] == 1.0f ||
            inputImage[CHECK_INDEX(colIndex - 1 + rowIndex * imageWidth, imageWidth, imageHeight, index)] == 1.0f ||
            inputImage[CHECK_INDEX(colIndex + (rowIndex + 1) * imageWidth, imageWidth, imageHeight, index)] == 1.0f ||
            inputImage[CHECK_INDEX(colIndex + (rowIndex - 1) * imageWidth, imageWidth, imageHeight, index)] == 1.0f ||
            inputImage[CHECK_INDEX(colIndex - 1 + (rowIndex - 1) * imageWidth, imageWidth, imageHeight, index)] ==
            1.0f ||
            inputImage[CHECK_INDEX(colIndex + 1 + (rowIndex + 1) * imageWidth, imageWidth, imageHeight, index)] ==
            1.0f ||
            inputImage[CHECK_INDEX(colIndex - 1 + (rowIndex - 1) * imageWidth, imageWidth, imageHeight, index)] ==
            1.0f ||
            inputImage[CHECK_INDEX(colIndex + 1 + (rowIndex - 1) * imageWidth, imageWidth, imageHeight, index)] ==
            1.0f ||
            inputImage[CHECK_INDEX(colIndex - 1 + (rowIndex + 1) * imageWidth, imageWidth, imageHeight, index)] == 1.0f
    )) {
        outputImage[colIndex + rowIndex * imageWidth] = WEAK_EDGE_VALUE;
        return;
    }

    outputImage[colIndex + rowIndex * imageWidth] = 0.0f;
}