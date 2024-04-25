constant sampler_t
sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_MIRRORED_REPEAT | CLK_FILTER_NEAREST;

#define GTE_WITH_EPSILON(a, b) (a >= b - 0.0001f)
#define LT_WITH_EPSILON(a, b) (a < b + 0.0001f)
#define IN_RANGE_WITH_EPSILON(a, b, c) (GTE_WITH_EPSILON(a, b) && LT_WITH_EPSILON(a, c))

__kernel void grayscale_image(
        read_only image2d_t inputImage,
        write_only image2d_t outputImage
){
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    int2 imageSize = (int2)(get_image_width(inputImage), get_image_height(inputImage));
    if (coord.x >= imageSize.x || coord.y >= imageSize.y) {
        return;
    }

    float4 pixel = read_imagef(inputImage, sampler, coord);
    float sum = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;

    write_imagef(outputImage, coord, (float4)(sum, 0.0f, 0.0f, 0.0f));
}

__kernel void gaussian_blur(
        read_only image2d_t inputImage,
        write_only image2d_t outputImage
){
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    int2 imageSize = (int2)(get_image_width(inputImage), get_image_height(inputImage));
    if (coord.x >= imageSize.x || coord.y >= imageSize.y) {
        return;
    }

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
        __global float *intensityImage,
        __global float *orientationImage
) {
    int2 coords = (int2)(get_global_id(0), get_global_id(1));
    int2 imageSize = (int2)(get_image_width(inputImage), get_image_height(inputImage));
    if (coords.x >= imageSize.x || coords.y >= imageSize.y) {
        return;
    }

    const float sobelKernelX[3][3] = {
            {-1.0f, 0.0f, 1.0f},
            {-2.0f, 0.0f, 2.0f},
            {-1.0f, 0.0f, 1.0f}
    };
    const float sobelKernelY[3][3] = {
            {1.0f,  2.0f,  1.0f},
            {0.0f,  0.0f,  0.0f},
            {-1.0f, -2.0f, -1.0f}
    };

    float sobelX = 0.0f;
    float sobelY = 0.0f;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int2 neighborCoord = (int2)(coords.x + j, coords.y + i);
            sobelX += read_imagef(inputImage, sampler, neighborCoord).x * sobelKernelX[i + 1][j + 1];
            sobelY += read_imagef(inputImage, sampler, neighborCoord).x * sobelKernelY[i + 1][j + 1];
        }
    }

    float sobelMagnitude = sqrt(sobelX * sobelX + sobelY * sobelY);

    float orientation = atan2(sobelY, sobelX);
    orientation = orientation * (180.0f / M_PI);
    orientation = round(orientation / 22.5f) * 22.5f;
    orientation = fmod(orientation + 180.0f, 180.0f);
    if (orientation < 0.0f) {
        orientation += 180.0f;
    }

    intensityImage[coords.x + coords.y * get_global_size(0)] = sobelMagnitude;
    orientationImage[coords.x + coords.y * get_global_size(0)] = orientation;
}

#define CHECK_INDEX(index, imageWidth, imageHeight, fallback) ((index >= 0 && index < imageWidth * imageHeight) ? index : fallback)

__kernel void edge_thinning(
        __global float *intensityImage,
        __global float *orientationImage,
        __global float *outputBuffer
) {
    int2 coords = (int2)(get_global_id(0), get_global_id(1));
     
    float q = 255.0f;
    float r = 255.0f;

    int angleIndex = coords.x + coords.y * get_global_size(0);
    float angle = orientationImage[angleIndex];

#define READ_ORIENTATION_WITH_FALLBACK(x, y) (intensityImage[CHECK_INDEX(x + y * get_global_size(0), get_global_size(0), get_global_size(1), angleIndex)])

    if (IN_RANGE_WITH_EPSILON(angle, 0.0f, 22.5f)) {
        q = READ_ORIENTATION_WITH_FALLBACK(coords.x + 1, coords.y);
        r = READ_ORIENTATION_WITH_FALLBACK(coords.x - 1, coords.y);
    } else if (IN_RANGE_WITH_EPSILON(angle, 22.5f, 67.5f)) {
        q = READ_ORIENTATION_WITH_FALLBACK(coords.x - 1, coords.y + 1);
        r = READ_ORIENTATION_WITH_FALLBACK(coords.x + 1, coords.y - 1);
    } else if (IN_RANGE_WITH_EPSILON(angle, 67.5f, 112.5f)) {
        q = READ_ORIENTATION_WITH_FALLBACK(coords.x, coords.y + 1);
        r = READ_ORIENTATION_WITH_FALLBACK(coords.x, coords.y - 1);
    } else if (IN_RANGE_WITH_EPSILON(angle, 112.5f, 157.5f)) {
        q = READ_ORIENTATION_WITH_FALLBACK(coords.x - 1, coords.y - 1);
        r = READ_ORIENTATION_WITH_FALLBACK(coords.x + 1, coords.y + 1);
    } else {
        q = READ_ORIENTATION_WITH_FALLBACK(coords.x + 1, coords.y);
        r = READ_ORIENTATION_WITH_FALLBACK(coords.x - 1, coords.y);
    }

    float intensity = intensityImage[coords.x + coords.y * get_global_size(0)];
    if (GTE_WITH_EPSILON(intensity, q) && GTE_WITH_EPSILON(intensity, r)) {
        outputBuffer[coords.x + coords.y * get_global_size(0)] = intensity;
    } else {
        outputBuffer[coords.x + coords.y * get_global_size(0)] = 0.00f;
    }
}

__kernel void find_max_intensity(
        __global float *intensityBuffer,
        __global float *result,
        const int bufferSize
) {
    int globalIndex = get_global_id(0);

    float localMax = intensityBuffer[globalIndex];

    for (int i = globalIndex + get_global_size(0); i < bufferSize; i += get_global_size(0)) {
        localMax = max(localMax, intensityBuffer[i]);
    }

    __local float localMaxBuffer[256];
    localMaxBuffer[get_local_id(0)] = localMax;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
        if (get_local_id(0) < stride) {
            localMaxBuffer[get_local_id(0)] = max(localMaxBuffer[get_local_id(0)],
                                                  localMaxBuffer[get_local_id(0) + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (get_local_id(0) == 0) {
        result[get_group_id(0)] = localMaxBuffer[0];
    }
}

#define STRONG_EDGE_VALUE 1.0f
#define WEAK_EDGE_VALUE 0.33f

#define LOW_THRESHOLD_RATIO 0.727f
#define HIGH_THRESHOLD_RATIO 0.765f

__kernel void double_thresholding(
        __global float *inputImage,
        write_only image2d_t outputImage,
        const float maxIntensityValue
) {
    int2 coords = (int2)(get_global_id(0), get_global_id(1));
    int2 imageSize = (int2)(get_image_width(outputImage), get_image_height(outputImage));
    if (coords.x >= imageSize.x || coords.y >= imageSize.y) {
        return;
    }

    float high_threshold = maxIntensityValue * HIGH_THRESHOLD_RATIO;
    float low_threshold = high_threshold * LOW_THRESHOLD_RATIO;

    float intensity = inputImage[coords.x + coords.y * get_global_size(0)];
    if (GTE_WITH_EPSILON(intensity, high_threshold)) {
        write_imagef(outputImage, coords, (float4)(STRONG_EDGE_VALUE, 0.0f, 0.0f, 0.0f));
    } else if (GTE_WITH_EPSILON(intensity, low_threshold)) {
        write_imagef(outputImage, coords, (float4)(WEAK_EDGE_VALUE, 0.0f, 0.0f, 0.0f));
    } else {
        write_imagef(outputImage, coords, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
    }
}


__kernel void edge_histeresis(
        read_only image2d_t inputImage,
        write_only image2d_t outputImage
) {
    int2 coords = (int2)(get_global_id(0), get_global_id(1));
    int2 imageSize = (int2)(get_image_width(inputImage), get_image_height(inputImage));
    if (coords.x >= imageSize.x || coords.y >= imageSize.y) {
        return;
    }

    float intensity = read_imagef(inputImage, sampler, coords).x;

    if (intensity == STRONG_EDGE_VALUE) {
        write_imagef(outputImage, coords, (float4)(STRONG_EDGE_VALUE, 0.0f, 0.0f, 0.0f));
        return;
    }

    if (intensity == WEAK_EDGE_VALUE && (
            read_imagef(inputImage, sampler, (int2)(coords.x + 1, coords.y)).x == STRONG_EDGE_VALUE ||
            read_imagef(inputImage, sampler, (int2)(coords.x - 1, coords.y)).x == STRONG_EDGE_VALUE ||
            read_imagef(inputImage, sampler, (int2)(coords.x, coords.y + 1)).x == STRONG_EDGE_VALUE ||
            read_imagef(inputImage, sampler, (int2)(coords.x, coords.y - 1)).x == STRONG_EDGE_VALUE ||
            read_imagef(inputImage, sampler, (int2)(coords.x - 1, coords.y - 1)).x == STRONG_EDGE_VALUE ||
            read_imagef(inputImage, sampler, (int2)(coords.x + 1, coords.y + 1)).x == STRONG_EDGE_VALUE ||
            read_imagef(inputImage, sampler, (int2)(coords.x - 1, coords.y - 1)).x == STRONG_EDGE_VALUE ||
            read_imagef(inputImage, sampler, (int2)(coords.x + 1, coords.y - 1)).x == STRONG_EDGE_VALUE ||
            read_imagef(inputImage, sampler, (int2)(coords.x - 1, coords.y + 1)).x == STRONG_EDGE_VALUE)) {
        write_imagef(outputImage, coords, (float4)(STRONG_EDGE_VALUE, 0.0f, 0.0f, 0.0f));
        return;
    }

    write_imagef(outputImage, coords, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
}