#include <stdio.h>
#include <CL/opencl.h>
#include <assert.h>
#include <string.h>
#include "lodepng.h"

float *convertToFloatArray(const uint8_t *array, size_t size) {
    float *result = (float *) malloc(size * sizeof(float));
    for (size_t i = 0; i < size; i++) {
        result[i] = (float) array[i] / 255.0f;
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

    uint32_t error = lodepng_decode24_file(&imageBuffer, &width, &height, "lenna.png");
    assert(error == 0);

    printf("Image width: %d height: %d\n", width, height);
    float *imageFloatBuffer = convertToFloatArray(imageBuffer, width * height * 3);

    cl_device_id device = getOpenCLDevice();
    cl_context context = createOpenCLContext(device);

    cl_int commandQueueResult;
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &commandQueueResult);
    assert(commandQueueResult == CL_SUCCESS);

    const char *programSource = loadProgramSource("compute.cl");

    cl_int imageResult;
    cl_mem image = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * 3 * sizeof(float), nullptr,
                                  &imageResult);
    assert(imageResult == CL_SUCCESS);
    cl_int enqueueResult = clEnqueueWriteBuffer(queue, image, CL_TRUE, 0, width * height * 3 * sizeof(float),
                                                imageFloatBuffer, 0, nullptr, nullptr);
    assert(enqueueResult == CL_SUCCESS);

    cl_int outputImageResult;
    cl_mem grayscaleOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * sizeof(float), nullptr,
                                                  &outputImageResult);
    assert(outputImageResult == CL_SUCCESS);
    cl_mem gaussianOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * sizeof(float), nullptr,
                                                 &outputImageResult);
    assert(outputImageResult == CL_SUCCESS);
    cl_mem sobelOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * width * height * sizeof(float), nullptr,
                                              &outputImageResult);
    assert(outputImageResult == CL_SUCCESS);
    cl_mem edgeThinningOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * sizeof(float),
                                                     nullptr,
                                                     &outputImageResult);
    assert(outputImageResult == CL_SUCCESS);

    cl_kernel grayscaleKernel = createOpenCLKernel(context, device, programSource, "grayscale_image");
    cl_int err = clSetKernelArg(grayscaleKernel, 0, sizeof(cl_mem), &image);
    err |= clSetKernelArg(grayscaleKernel, 1, sizeof(cl_mem), &grayscaleOutputBuffer);
    assert(err == CL_SUCCESS);

    cl_kernel gaussian = createOpenCLKernel(context, device, programSource, "gaussian_blur");
    err = clSetKernelArg(gaussian, 0, sizeof(cl_mem), &grayscaleOutputBuffer);
    err |= clSetKernelArg(gaussian, 1, sizeof(cl_mem), &gaussianOutputBuffer);
    assert(err == CL_SUCCESS);

    cl_kernel sobel = createOpenCLKernel(context, device, programSource, "sobel_filter");
    err = clSetKernelArg(sobel, 0, sizeof(cl_mem), &gaussianOutputBuffer);
    err |= clSetKernelArg(sobel, 1, sizeof(cl_mem), &sobelOutputBuffer);
    assert(err == CL_SUCCESS);

    cl_kernel edge_thinning = createOpenCLKernel(context, device, programSource, "edge_thinning");
    err = clSetKernelArg(edge_thinning, 0, sizeof(cl_mem), &sobelOutputBuffer);
    err |= clSetKernelArg(edge_thinning, 1, sizeof(cl_mem), &edgeThinningOutputBuffer);
    assert(err == CL_SUCCESS);


    size_t globalWorkSize[2] = {width, height};
    size_t localWorkSize[2] = {16, 16};
    cl_int kernelEnqueueResult = clEnqueueNDRangeKernel(queue, grayscaleKernel, 2, nullptr, globalWorkSize,
                                                        localWorkSize, 0,
                                                        nullptr, nullptr);
    kernelEnqueueResult |= clEnqueueNDRangeKernel(queue, gaussian, 2, nullptr, globalWorkSize, localWorkSize, 0,
                                                  nullptr, nullptr);
    kernelEnqueueResult |= clEnqueueNDRangeKernel(queue, sobel, 2, nullptr, globalWorkSize, localWorkSize, 0,
                                                  nullptr, nullptr);
    kernelEnqueueResult |= clEnqueueNDRangeKernel(queue, edge_thinning, 2, nullptr, globalWorkSize, localWorkSize, 0,
                                                  nullptr, nullptr);
    assert(kernelEnqueueResult == CL_SUCCESS);

    clFinish(queue);

    float *outputImageBuffer = (float *) malloc(width * height * sizeof(float));
    cl_int readResult = clEnqueueReadBuffer(queue, edgeThinningOutputBuffer, CL_TRUE, 0, width * height * sizeof(float),
                                            outputImageBuffer, 0, nullptr, nullptr);
    assert(readResult == CL_SUCCESS);

    uint8_t *outputImageByteArray = convertToByteArray(outputImageBuffer, width * height);
    error = lodepng_encode_file("/tmp/lena_out.png", outputImageByteArray, width, height, LCT_GREY, 8);
    assert(error == 0);


    free(outputImageByteArray);
    free(outputImageBuffer);
    free(imageBuffer);
    free(imageFloatBuffer);
    free(programSource);

    return 0;
}