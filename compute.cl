constant sampler_t
sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_MIRRORED_REPEAT | CLK_FILTER_NEAREST;

__kernel void grayscale_image(
        read_only image2d_t inputImage,
        write_only image2d_t outputImage
){
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float4 pixel = read_imagef(inputImage, sampler, coord);
    float sum = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;

    write_imagef(outputImage, coord, (float4)(sum, 0.0f, 0.0f, 0.0f));
}

__kernel void gaussian_blur(
        read_only image2d_t inputImage,
        write_only image2d_t outputImage
){
    int2 coord = (int2)(get_global_id(0), get_global_id(1));    

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
            int2 neighborCoord = (int2)(coord.x + i, coord.y + j);
            newPixelValue += read_imagef(inputImage, sampler, neighborCoord).x * gaussianTable[(i + 2) * 3 + (j + 2)];
        }
    }

    newPixelValue = clamp(newPixelValue, 0.0f, 1.0f);
    write_imagef(outputImage, coord, (float4)(newPixelValue, 0.0f, 0.0f, 0.0f));
}

__kernel void sobel_filter(
        read_only image2d_t inputImage,
        write_only image2d_t intensityImage,
        write_only image2d_t orientationImage
) {
    int2 coords = (int2)(get_global_id(0), get_global_id(1));

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
            int2 neighborCoord = (int2)(coords.x + i, coords.y + j);
            sobelX += read_imagef(inputImage, sampler, neighborCoord).x * sobelKernelX[(i + 1) * 3 + (j + 1)];
            sobelY += read_imagef(inputImage, sampler, neighborCoord).x * sobelKernelY[(i + 1) * 3 + (j + 1)];
        }
    }

    float sobelMagnitude = sqrt(sobelX * sobelX + sobelY * sobelY);
    float orientation = atan2(sobelY, sobelX);
    orientation = orientation * (180.0f / M_PI);
    float orientationRounded = round(orientation / 45.0f) * 45.0f;
    orientation = fmod(orientationRounded + 180.0f, 180.0f);

    sobelMagnitude = clamp(sobelMagnitude, 0.0f, 1.0f);
    orientation = clamp(orientation, 0.0f, 180.0f);

    write_imagef(intensityImage, coords, (float4)(sobelMagnitude, 0.0f, 0.0f, 0.0f));
    write_imagef(orientationImage, coords, (float4)(orientation, 0.0f, 0.0f, 0.0f));
}

__kernel void edge_thinning(
        read_only image2d_t intensityImage,
        read_only image2d_t orientationImage,
        write_only image2d_t outputImage
) {
    int2 coords = (int2)(get_global_id(0), get_global_id(1));
     
    float q = 255.0f;
    float r = 255.0f;

    float angle = read_imagef(orientationImage, sampler, coords).x;

    if (angle >= 0 && angle < 22.5) {
        q = read_imagef(intensityImage, sampler, (int2)(coords.x + 1, coords.y)).x;
        r = read_imagef(intensityImage, sampler, (int2)(coords.x - 1, coords.y)).x;
    } else  if(angle >= 22.5 && angle < 67.5) {
        q = read_imagef(intensityImage, sampler, (int2)(coords.x + 1, coords.y - 1)).x;
        r = read_imagef(intensityImage, sampler, (int2)(coords.x - 1, coords.y + 1)).x;
    } else if(angle >= 67.5 && angle < 112.5) {
        q = read_imagef(intensityImage, sampler, (int2)(coords.x, coords.y - 1)).x;
        r = read_imagef(intensityImage, sampler, (int2)(coords.x, coords.y + 1)).x;
    } else if(angle >= 112.5 && angle < 157.5) {
        q = read_imagef(intensityImage, sampler, (int2)(coords.x - 1, coords.y - 1)).x;
        r = read_imagef(intensityImage, sampler, (int2)(coords.x + 1, coords.y + 1)).x;
    } else if(angle >= 157.5 && angle <= 180.0) {
        q = read_imagef(intensityImage, sampler, (int2)(coords.x - 1, coords.y)).x;
        r = read_imagef(intensityImage, sampler, (int2)(coords.x + 1, coords.y)).x;
    }

    float intensity = read_imagef(intensityImage, sampler, coords).x;
    if (intensity >= q && intensity >= r) {
        write_imagef(outputImage, coords, (float4)(intensity, 0.0f, 0.0f, 0.0f));
    } else {
        write_imagef(outputImage, coords, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
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