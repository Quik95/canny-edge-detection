#include <stdio.h>
#include <CL/opencl.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include "lodepng.h"

#define GET_IMAGE_SIZE(width, height) (4 * width * height)

float *convertToFloatArray(const uint8_t *array, size_t size) {
    float *result = (float *) malloc(size * sizeof(float));
    for (size_t i = 0; i < size; i += 4) {
        result[i] = (float) array[i] / 255.0f;
        result[i + 1] = (float) array[i + 1] / 255.0f;
        result[i + 2] = (float) array[i + 2] / 255.0f;
        result[i + 3] = 1.0f;
    }
    return result;
}

uint8_t *convertToByteArray(const float *array, size_t size) {
    uint8_t *result = (uint8_t *) malloc(size * sizeof(uint8_t));
    for (size_t i = 0; i < size; i += 4) {
        result[i] = (uint8_t)(array[i] * 255.0f);
        result[i + 1] = (uint8_t)(array[i + 1] * 255.0f);
        result[i + 2] = (uint8_t)(array[i + 2] * 255.0f);
        result[i + 3] = 255;
    }
    return result;
}

cl_device_id getOpenCLDevice() {
    cl_platform_id platform[64];
    uint32_t platformCount;
    cl_int platformResult = clGetPlatformIDs(64, platform, &platformCount);
    assert(platformResult == CL_SUCCESS);

    cl_device_id device = nullptr;
    for (int i = 0; i < platformCount && device == nullptr; i++) {
        cl_device_id devices[64];
        uint32_t deviceCount;
        cl_int deviceResult = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_GPU, 64, devices, &deviceCount);

        if (deviceResult != CL_SUCCESS) {
            continue;
        }

        for (int j = 0; j < deviceCount; j++) {
            char vendorName[256];
            size_t vendorNameLength;
            cl_int deviceInfoResult = clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 256, vendorName, &vendorNameLength);
            if (deviceInfoResult == CL_SUCCESS) {
                device = devices[j];
            }
        }
    }

    printf("Device: %p\n", device);
    return device;
}

cl_context createOpenCLContext(cl_device_id device) {
    cl_int contextResult;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &contextResult);
    assert(contextResult == CL_SUCCESS);

    return context;
}

cl_kernel createOpenCLKernel(cl_context context, cl_device_id device, const char *source, const char *kernelName) {
    size_t length = strlen(source);
    cl_int programResult;
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) &source, &length, &programResult);
    assert(programResult == CL_SUCCESS);

    cl_int programBuildResult = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
    if (programBuildResult != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        char *log = (char *) malloc(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);
        printf("Build log: %s\n", log);
        free(log);
    }

    cl_int kernelResult;
    cl_kernel kernel = clCreateKernel(program, kernelName, &kernelResult);
    assert(kernelResult == CL_SUCCESS);
    return kernel;
}

const char *loadProgramSource(const char *filename) {
    FILE *file = fopen(filename, "r");
    assert(file != nullptr);
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    rewind(file);
    char *programSource = (char *) malloc(fileSize + 1);
    bzero(programSource, fileSize + 1);
    int n = fread(programSource, 1, fileSize, file);
    assert(n > 0);
    return programSource;
}

int main() {
    printf("We are running OpenCL code\n");

    uint8_t *imageBuffer;
    uint32_t width, height;

    uint32_t error = lodepng_decode32_file(&imageBuffer, &width, &height, "lenna.png");
    assert(error == 0);

    printf("Image width: %d height: %d\n", width, height);
    float *imageFloatBuffer = convertToFloatArray(imageBuffer, GET_IMAGE_SIZE(width, height));

    clock_t start = clock();
    cl_device_id device = getOpenCLDevice();
    cl_context context = createOpenCLContext(device);

    cl_int commandQueueResult;
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &commandQueueResult);
    assert(commandQueueResult == CL_SUCCESS);

    const char *programSource = loadProgramSource("compute.cl");

    cl_image_format imageFormat = {.image_channel_data_type = CL_FLOAT, .image_channel_order = CL_RGBA};
    cl_image_format grayscaleImageFormat = {.image_channel_data_type = CL_FLOAT, .image_channel_order = CL_INTENSITY};
    cl_image_desc imageDesc = {.image_type = CL_MEM_OBJECT_IMAGE2D, .image_width = width, .image_height = height};

    cl_int imageResult;
    cl_mem colorImageBuffer = clCreateImage(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &imageFormat, &imageDesc,
                                            imageFloatBuffer, &imageResult);
    printf("image result: %d\n", imageResult);
    assert(imageResult == CL_SUCCESS);

    float *grayscaleImageBuffer = (float *) malloc(width * height * sizeof(float));
    cl_mem auxiliaryImageBuffer = clCreateImage(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &grayscaleImageFormat,
                                                &imageDesc, grayscaleImageBuffer, &imageResult);
    assert(imageResult == CL_SUCCESS);

    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {imageDesc.image_width, imageDesc.image_height, 1};

    cl_int err;
    cl_kernel grayscaleKernel = createOpenCLKernel(context, device, programSource, "grayscale_image");
    err = clSetKernelArg(grayscaleKernel, 0, sizeof(cl_mem), &colorImageBuffer);
    err |= clSetKernelArg(grayscaleKernel, 1, sizeof(cl_mem), &auxiliaryImageBuffer);
    assert(err == CL_SUCCESS);

//    cl_kernel gaussian = createOpenCLKernel(context, device, programSource, "gaussian_blur");
//    err = clSetKernelArg(gaussian, 0, sizeof(cl_mem), &auxiliaryBuffer);
//    err |= clSetKernelArg(gaussian, 1, sizeof(cl_mem), &colorImageBuffer);
//    assert(err == CL_SUCCESS);
//
//    cl_kernel sobel = createOpenCLKernel(context, device, programSource, "sobel_filter");
//    err = clSetKernelArg(sobel, 0, sizeof(cl_mem), &colorImageBuffer);
//    err |= clSetKernelArg(sobel, 1, sizeof(cl_mem), &auxiliaryBuffer);
//    assert(err == CL_SUCCESS);
//
//    cl_kernel edge_thinning = createOpenCLKernel(context, device, programSource, "edge_thinning");
//    err = clSetKernelArg(edge_thinning, 0, sizeof(cl_mem), &auxiliaryBuffer);
//    err |= clSetKernelArg(edge_thinning, 1, sizeof(cl_mem), &colorImageBuffer);
//    assert(err == CL_SUCCESS);
//
//    cl_kernel double_thresholding = createOpenCLKernel(context, device, programSource, "double_thresholding");
//    err = clSetKernelArg(double_thresholding, 0, sizeof(cl_mem), &colorImageBuffer);
//    err |= clSetKernelArg(double_thresholding, 1, sizeof(cl_mem), &auxiliaryBuffer);
//    assert(err == CL_SUCCESS);
//
//    cl_kernel edge_histeresis = createOpenCLKernel(context, device, programSource, "edge_histeresis");
//    err = clSetKernelArg(edge_histeresis, 0, sizeof(cl_mem), &auxiliaryBuffer);
//    err |= clSetKernelArg(edge_histeresis, 1, sizeof(cl_mem), &colorImageBuffer);
//    assert(err == CL_SUCCESS);


    size_t globalWorkSize[2] = {width, height};
    cl_int kernelEnqueueResult = clEnqueueNDRangeKernel(queue, grayscaleKernel, 2, nullptr, globalWorkSize,
                                                        nullptr, 0,
                                                        nullptr, nullptr);
//    kernelEnqueueResult |= clEnqueueNDRangeKernel(queue, gaussian, 2, nullptr, globalWorkSize, nullptr, 0,
//                                                  nullptr, nullptr);
//    kernelEnqueueResult |= clEnqueueNDRangeKernel(queue, sobel, 2, nullptr, globalWorkSize, nullptr, 0,
//                                                  nullptr, nullptr);
//    kernelEnqueueResult |= clEnqueueNDRangeKernel(queue, edge_thinning, 2, nullptr, globalWorkSize, nullptr, 0,
//                                                  nullptr, nullptr);
//    kernelEnqueueResult |= clEnqueueNDRangeKernel(queue, double_thresholding, 2, nullptr, globalWorkSize, nullptr,
//                                                  0,
//                                                  nullptr, nullptr);
//    kernelEnqueueResult |= clEnqueueNDRangeKernel(queue, edge_histeresis, 2, nullptr, globalWorkSize, nullptr, 0,
//                                                  nullptr, nullptr);
    assert(kernelEnqueueResult == CL_SUCCESS);


    float *outputImageBuffer = (float *) malloc(width * height * sizeof(float));
//    float *outputImageBuffer = convertToFloatArray(auxiliaryBuffer, 3 * width * height);
//    cl_int readResult = clEnqueueReadBuffer(queue, colorImageBuffer, CL_TRUE, 0,
//                                            width * height * sizeof(float),
//                                            outputImageBuffer, 0, nullptr, nullptr);
//    cl_int readResult = clEnqueueReadBuffer(queue, auxiliaryImageBuffer, CL_TRUE, 0,
//                                            3 * width * height * sizeof(float),
//                                            outputImageBuffer, 0, nullptr, nullptr);

//    assert(readResult == CL_SUCCESS);

    err = clEnqueueReadImage(queue, auxiliaryImageBuffer, CL_TRUE, origin, region, 0, 0, outputImageBuffer, 0, nullptr,
                             nullptr);
    assert(err == CL_SUCCESS);

    start = clock();
    clFinish(queue);
    clock_t end = clock();
    printf("Compute done in %f miliseconds\n", (double) (end - start) / CLOCKS_PER_SEC * 1000);

    uint8_t *outputImageByteArray = malloc(width * height * sizeof(uint8_t));
    for (size_t i = 0; i < width * height; i++) {
        outputImageByteArray[i] = (uint8_t)(outputImageBuffer[i] * 255.0f);
    }
    error = lodepng_encode_file("/tmp/lena_out.png", outputImageByteArray, width, height, LCT_GREY, 8);
    assert(error == 0);


//    free(outputImageByteArray);
    free(outputImageBuffer);
    free(imageBuffer);
//    free(imageFloatBuffer);
    free(programSource);

    return 0;
}