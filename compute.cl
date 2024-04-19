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

__kernel void sobel_filter(
    __global float* inputImage,
    __global float* outputImage
) {
    int colIndex = get_global_id(0);
    int rowIndex = get_global_id(1);
    int imageWidth = get_global_size(0);

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

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int neighborCol = colIndex + j;
            int neighborRow = rowIndex + i;

            if (neighborCol >= 0 && neighborCol < imageWidth && neighborRow >= 0) {
                int neighborIndex = neighborRow * imageWidth + neighborCol;
                sobelX += inputImage[neighborIndex] * sobelKernelX[(i + 1) * 3 + (j + 1)];
                sobelY += inputImage[neighborIndex] * sobelKernelY[(i + 1) * 3 + (j + 1)];
            }
        }
    }

    float sobelMagnitude = sqrt(sobelX * sobelX + sobelY * sobelY);
    float orientation = atan2(sobelY, sobelX);
    orientation = orientation * (180.0f / M_PI);
    float orientationRounded = round(orientation / 45.0f) * 45.0f;
    orientation = fmod(orientationRounded + 180.0f, 180.0f);

    outputImage[rowIndex * imageWidth + colIndex] = clamp(sobelMagnitude, 0.0f, 1.0f);
    outputImage[rowIndex * imageWidth + colIndex + imageWidth * imageWidth] = clamp(orientation, 0.0f, 180.0f);
}

__kernel void edge_thinning(
    __global float* inputImage,
    __global float* outputImage
) {
    int colIndex = get_global_id(0);
    int rowIndex = get_global_id(1);
    int imageWidth = get_global_size(0);

    float q = 255.0f;
    float r = 255.0f;

    int angleIndex = rowIndex * imageWidth + colIndex + imageWidth * imageWidth;
    float angle = inputImage[angleIndex];

    if (angle >= 0 && angle < 22.5) {
        q = inputImage[rowIndex * imageWidth + colIndex + 1];
        r=  inputImage[rowIndex * imageWidth + colIndex - 1];
    } else  if(angle >= 22.5 && angle < 67.5) {
        q = inputImage[(rowIndex - 1) * imageWidth + colIndex - 1];
        r = inputImage[(rowIndex + 1) * imageWidth + colIndex + 1];
    } else if(angle >= 67.5 && angle < 112.5) {
        q = inputImage[(rowIndex - 1) * imageWidth + colIndex];
        r = inputImage[(rowIndex + 1) * imageWidth + colIndex];
    } else if(angle >= 112.5 && angle < 157.5) {
        q = inputImage[(rowIndex - 1) * imageWidth + colIndex + 1];
        r = inputImage[(rowIndex + 1) * imageWidth + colIndex - 1];
    } else if(angle >= 157.5 && angle <= 180.0) {
        q = inputImage[rowIndex * imageWidth + colIndex - 1];
        r = inputImage[rowIndex * imageWidth + colIndex + 1];
    }

    float intensity = inputImage[rowIndex * imageWidth + colIndex];
    if (intensity >= q && intensity >= r) {
        outputImage[rowIndex * imageWidth + colIndex] = intensity;
    } else {
        outputImage[rowIndex * imageWidth + colIndex] = 0.0f;
    }
}

__kernel void double_thresholding(
    __global float* inputImage,
    __global float* outputImage
) {
    int colIndex = get_global_id(0);
    int rowIndex = get_global_id(1);
    int imageWidth = get_global_size(0);

    float highThreshold = 0.2f;
    float lowThreshold = 0.1f;

    float strongEdgePixel = 1.0f;
    float weakEdgePixel = 0.5f;

    float intensity = inputImage[rowIndex * imageWidth + colIndex];
    if (intensity >= highThreshold) {
        outputImage[rowIndex * imageWidth + colIndex] = strongEdgePixel;
    } else if (intensity >= lowThreshold) {
        outputImage[rowIndex * imageWidth + colIndex] = weakEdgePixel;
    } else {
        outputImage[rowIndex * imageWidth + colIndex] = 0.0f;
    }
}