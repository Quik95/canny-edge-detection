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
    for (size_t i = 0; i < size; i++) {
        result[i] = (uint8_t) (array[i] * 255.0f);
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

cl_program createOpenCLProgram(cl_context context, cl_device_id device, const char *source) {
    size_t length = strlen(source);
    cl_int programResult;
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) &source, &length, &programResult);
    assert(programResult == CL_SUCCESS);

    cl_int programBuildResult = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
    if (programBuildResult != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        char *log = (char *) malloc(logSize + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);
        log[logSize] = '\0';
        printf("Build log: %s\n", log);
        free(log);
    }

    return program;
}

cl_kernel createOpenCLKernel(cl_program program, const char *kernelName) {
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

int main(int argc, char **argv) {
    printf("We are running OpenCL code\n");

    char inputImagePath[1024] = {0};
    char outputImagePath[1024] = {0};
    char kernelPath[1024] = {0};

    for (int i = 0; i < argc; i++) {
        printf("Argument %d: %s\n", i, argv[i]);
    }

    switch (argc) {
        case 2:
            strcpy(inputImagePath, "lenna.png");
            strcpy(outputImagePath, "/tmp/lena_out.png");
            break;
        case 3:
            assert(strstr(argv[2], ".png") != nullptr);
            strcpy(inputImagePath, argv[2]);
            strcpy(outputImagePath, "/tmp/lena_out.png");
            break;
        case 4:
            assert(strstr(argv[3], ".png") != nullptr);
            assert(strstr(argv[4], ".png") != nullptr);
            strcpy(inputImagePath, argv[2]);
            strcpy(outputImagePath, argv[3]);
            break;
        default:
            printf("Usage: %s [input_image_path] [output_image_path]\n", argv[0]);
            return 1;
    }
    strcpy(kernelPath, argv[1]);

    const char *programSource = loadProgramSource(kernelPath);
    printf("Input image path: %s\n", inputImagePath);
    printf("Output image path: %s\n", outputImagePath);

    uint8_t *imageBuffer;
    uint32_t width, height;

    uint32_t error = lodepng_decode32_file(&imageBuffer, &width, &height, inputImagePath);
    assert(error == 0);

    printf("Image width: %d height: %d\n", width, height);
    float *imageFloatBuffer = convertToFloatArray(imageBuffer, GET_IMAGE_SIZE(width, height));

    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    cl_device_id device = getOpenCLDevice();
    cl_context context = createOpenCLContext(device);

    cl_int commandQueueResult;
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &commandQueueResult);
    cl_program program = createOpenCLProgram(context, device, programSource);
    assert(commandQueueResult == CL_SUCCESS);

    struct timespec deviceSetupEnd;
    clock_gettime(CLOCK_MONOTONIC, &deviceSetupEnd);

    cl_image_format imageFormat = {.image_channel_data_type = CL_FLOAT, .image_channel_order = CL_RGBA};
    cl_image_format grayscaleImageFormat = {.image_channel_data_type = CL_FLOAT, .image_channel_order = CL_INTENSITY};
    cl_image_desc imageDesc = {.image_type = CL_MEM_OBJECT_IMAGE2D, .image_width = width, .image_height = height};

    cl_int imageResult;
    cl_mem colorImageBuffer = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, &imageFormat, &imageDesc,
                                            imageFloatBuffer, &imageResult);
    printf("image result: %d\n", imageResult);
    assert(imageResult == CL_SUCCESS);

    float *grayscaleImageBuffer = (float *) malloc(width * height * sizeof(float));
    cl_mem auxiliaryImageBuffer = clCreateImage(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &grayscaleImageFormat,
                                                &imageDesc, grayscaleImageBuffer, &imageResult);
    assert(imageResult == CL_SUCCESS);

    float *grayscaleAuxiliaryBuffer = (float *) malloc(width * height * sizeof(float));
    cl_mem auxiliaryGrayscaleBuffer = clCreateImage(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                                    &grayscaleImageFormat,
                                                    &imageDesc, grayscaleAuxiliaryBuffer, &imageResult);
    assert(imageResult == CL_SUCCESS);

    float *sobelIntensityBuffer = (float *) malloc(width * height * sizeof(float));
    cl_mem sobelIntensityCLBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                                   width * height * sizeof(float), sobelIntensityBuffer, &imageResult);
    assert(imageResult == CL_SUCCESS);

    float *sobelOrientationBuffer = (float *) malloc(width * height * sizeof(float));
    cl_mem sobelOrientationCLBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                                     width * height * sizeof(float), sobelOrientationBuffer,
                                                     &imageResult);
    assert(imageResult == CL_SUCCESS);

    float *edgeThinningBuffer = (float *) malloc(width * height * sizeof(float));
    cl_mem edgeThinningCLBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                                 width * height * sizeof(float), edgeThinningBuffer, &imageResult);
    assert(imageResult == CL_SUCCESS);

    float *maxIntensityValueBuffer = (float *) malloc(sizeof(float));
    cl_mem maxIntensityValueCLBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                                      sizeof(float), maxIntensityValueBuffer, &imageResult);

    struct timespec memoryBuffersEnd;
    clock_gettime(CLOCK_MONOTONIC, &memoryBuffersEnd);

    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {imageDesc.image_width, imageDesc.image_height, 1};

    cl_int err;
    cl_kernel grayscaleKernel = createOpenCLKernel(program, "grayscale_image");
    err = clSetKernelArg(grayscaleKernel, 0, sizeof(cl_mem), &colorImageBuffer);
    err |= clSetKernelArg(grayscaleKernel, 1, sizeof(cl_mem), &auxiliaryImageBuffer);
    assert(err == CL_SUCCESS);

    cl_kernel gaussian = createOpenCLKernel(program, "gaussian_blur");
    err = clSetKernelArg(gaussian, 0, sizeof(cl_mem), &auxiliaryImageBuffer);
    err |= clSetKernelArg(gaussian, 1, sizeof(cl_mem), &auxiliaryGrayscaleBuffer);
    assert(err == CL_SUCCESS);

    cl_kernel sobel = createOpenCLKernel(program, "sobel_filter");
    err = clSetKernelArg(sobel, 0, sizeof(cl_mem), &auxiliaryGrayscaleBuffer);
    err |= clSetKernelArg(sobel, 1, sizeof(cl_mem), &sobelIntensityCLBuffer);
    err |= clSetKernelArg(sobel, 2, sizeof(cl_mem), &sobelOrientationCLBuffer);
    assert(err == CL_SUCCESS);

    cl_kernel edge_thinning = createOpenCLKernel(program, "edge_thinning");
    err | clSetKernelArg(edge_thinning, 0, sizeof(cl_mem), &sobelIntensityCLBuffer);
    err |= clSetKernelArg(edge_thinning, 1, sizeof(cl_mem), &sobelOrientationCLBuffer);
    err |= clSetKernelArg(edge_thinning, 2, sizeof(cl_mem), &edgeThinningCLBuffer);
    assert(err == CL_SUCCESS);

    cl_kernel max_intensity = createOpenCLKernel(program, "find_max_intensity");
    err = clSetKernelArg(max_intensity, 0, sizeof(cl_mem), &sobelIntensityCLBuffer);
    err |= clSetKernelArg(max_intensity, 1, sizeof(cl_mem), &maxIntensityValueCLBuffer);
    cl_int bufferSize = width * height;
    err |= clSetKernelArg(max_intensity, 2, sizeof(cl_int), &bufferSize);
    assert(err == CL_SUCCESS);

    cl_kernel double_thresholding = createOpenCLKernel(program, "double_thresholding");
    err = clSetKernelArg(double_thresholding, 0, sizeof(cl_mem), &edgeThinningCLBuffer);
    err |= clSetKernelArg(double_thresholding, 1, sizeof(cl_mem), &auxiliaryGrayscaleBuffer);
    assert(err == CL_SUCCESS);

    cl_kernel edge_histeresis = createOpenCLKernel(program, "edge_histeresis");
    err = clSetKernelArg(edge_histeresis, 0, sizeof(cl_mem), &auxiliaryGrayscaleBuffer);
    err |= clSetKernelArg(edge_histeresis, 1, sizeof(cl_mem), &auxiliaryImageBuffer);
    assert(err == CL_SUCCESS);

    size_t globalWorkSize[2] = {width, height};
    size_t maxIntensityKernelWorkSize = width * height;
//    size_t localWorkSize[2] = {0, 0};
    void *localWorkSize = nullptr;
    cl_int kernelEnqueueResult = clEnqueueNDRangeKernel(queue, grayscaleKernel, 2, nullptr, globalWorkSize,
                                                        localWorkSize, 0,
                                                        nullptr, nullptr);
    kernelEnqueueResult |= clEnqueueNDRangeKernel(queue, gaussian, 2, nullptr, globalWorkSize, localWorkSize, 0,
                                                  nullptr, nullptr);
    kernelEnqueueResult |= clEnqueueNDRangeKernel(queue, sobel, 2, nullptr, globalWorkSize, localWorkSize, 0,
                                                  nullptr, nullptr);
    kernelEnqueueResult |= clEnqueueNDRangeKernel(queue, edge_thinning, 2, nullptr, globalWorkSize, localWorkSize, 0,
                                                  nullptr, nullptr);
    kernelEnqueueResult |= clEnqueueNDRangeKernel(queue, max_intensity, 1, nullptr, &maxIntensityKernelWorkSize,
                                                  nullptr, 0, nullptr,
                                                  nullptr);

    cl_float maxIntensityValue;
    err = clEnqueueReadBuffer(queue, maxIntensityValueCLBuffer, CL_TRUE, 0, sizeof(cl_float), &maxIntensityValue, 0,
                              nullptr, nullptr);
    assert(err == CL_SUCCESS);

    clSetKernelArg(double_thresholding, 2, sizeof(cl_float), &maxIntensityValue);
    kernelEnqueueResult |= clEnqueueNDRangeKernel(queue, double_thresholding, 2, nullptr, globalWorkSize, localWorkSize,
                                                  0,
                                                  nullptr, nullptr);
    kernelEnqueueResult |= clEnqueueNDRangeKernel(queue, edge_histeresis, 2, nullptr, globalWorkSize, localWorkSize, 0,
                                                  nullptr, nullptr);
    if (kernelEnqueueResult != CL_SUCCESS) {
        printf("Error: %d\n", kernelEnqueueResult);
    }
    assert(kernelEnqueueResult == CL_SUCCESS);

    clFinish(queue);
    struct timespec kernelComputeEnd;
    clock_gettime(CLOCK_MONOTONIC, &kernelComputeEnd);

    float *outputImageBuffer = (float *) malloc(width * height * sizeof(float));
    err = clEnqueueReadImage(queue, auxiliaryImageBuffer, CL_TRUE, origin, region, 0, 0, outputImageBuffer, 0,
                             nullptr,
                             nullptr);
//    err = clEnqueueReadBuffer(queue, edgeThinningCLBuffer, CL_TRUE, 0, width * height * sizeof(float),
//                              outputImageBuffer, 0, nullptr, nullptr);
    assert(err == CL_SUCCESS);
    struct timespec imageCopyEnd;
    clock_gettime(CLOCK_MONOTONIC, &imageCopyEnd);

#define TIME_IN_SECONDS(start, end) ((double) (end.tv_sec - start.tv_sec) + (double) (end.tv_nsec - start.tv_nsec) / 1000000000)

    putc('\n', stdout);
    printf("Device setup: %.5f seconds\n", TIME_IN_SECONDS(start, deviceSetupEnd));
    printf("Memory buffers: %.5f seconds\n", TIME_IN_SECONDS(deviceSetupEnd, memoryBuffersEnd));
    printf("Kernel compute: %.5f seconds\n", TIME_IN_SECONDS(memoryBuffersEnd, kernelComputeEnd));
    printf("Image copy: %.5f seconds\n", TIME_IN_SECONDS(kernelComputeEnd, imageCopyEnd));
    printf("Total time: %.5f seconds\n", TIME_IN_SECONDS(start, imageCopyEnd));
    printf("Total time excluding device setup: %.5f seconds\n", TIME_IN_SECONDS(deviceSetupEnd, imageCopyEnd));

    uint8_t *outputImageByteArray = convertToByteArray(outputImageBuffer, width * height);
    error = lodepng_encode_file(outputImagePath, outputImageByteArray, width, height, LCT_GREY, 8);
    assert(error == 0);

    err = clReleaseMemObject(auxiliaryImageBuffer);
    err |= clReleaseMemObject(colorImageBuffer);
    err |= clReleaseMemObject(sobelIntensityCLBuffer);
    err |= clReleaseMemObject(sobelOrientationCLBuffer);
    err |= clReleaseMemObject(edgeThinningCLBuffer);
    err |= clReleaseMemObject(auxiliaryGrayscaleBuffer);
    err |= clReleaseMemObject(auxiliaryImageBuffer);
    err |= clReleaseCommandQueue(queue);
    err |= clReleaseKernel(grayscaleKernel);
    err |= clReleaseKernel(gaussian);
    err |= clReleaseKernel(sobel);
    err |= clReleaseKernel(edge_thinning);
    err |= clReleaseKernel(max_intensity);
    err |= clReleaseKernel(double_thresholding);
    err |= clReleaseKernel(edge_histeresis);
    err |= clReleaseProgram(program);
    err |= clReleaseContext(context);
    assert(err == CL_SUCCESS);

    free(sobelOrientationBuffer);
    free(sobelIntensityBuffer);
    free(edgeThinningBuffer);
    free(grayscaleImageBuffer);
    free(grayscaleAuxiliaryBuffer);

    free(maxIntensityValueBuffer);
    free(outputImageByteArray);
    free(outputImageBuffer);
    free(imageBuffer);
    free(imageFloatBuffer);
    free((void *) programSource);

    return 0;
}