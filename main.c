#include <stdio.h>
#include <malloc.h>
#include <stdint.h>
#include <float.h>
#include "vendor/lodepng/lodepng.h"
#include <math.h>
#include <omp.h>
#include <time.h>
#include <assert.h>

// #define WRITE_INTERMEDIATE_IMAGES

static uint8_t *float_array_to_uint8_array(const float *in, uint8_t *out, uint32_t width, uint32_t height) {
    for (uint32_t i = 0; i < width * height; i++) {
        out[i] = (uint8_t) (in[i] * 255.0f);
    }
    return out;
}

static uint32_t calculate_index_with_wrap_around(int x, int y, uint32_t width, uint32_t height) {
    if (x < 0) x = (int) width + x;
    if (y < 0) y = (int) height + y;
    if (x >= (int) width) x = x - (int) width;
    if (y >= (int) height) y = y - (int) height;
    return y * width + x;
}

#ifdef WRITE_INTERMEDIATE_IMAGES

static void write_intermediate_image(const char *filename, const float *image, uint32_t width, uint32_t height) {
    uint8_t *image_uint8 = malloc(width * height * sizeof(uint8_t));
    image_uint8 = float_array_to_uint8_array(image, image_uint8, width, height);
    unsigned error = lodepng_encode_file(filename, image_uint8, width, height, LCT_GREY, 8);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    free(image_uint8);
}

#endif

#define LOW_THRESHOLD_RATIO 0.027f
#define HIGH_THRESHOLD_RATIO 0.064f
#define WEAK_EDGE_PIXEL 0.33f
#define STRONG_EDGE_PIXEL 1.0f

float *apply_gaussian_filter(float *image, uint32_t width, uint32_t height);

float *convert_to_grayscale(float *image, uint32_t width, uint32_t height);

float *apply_sobel_filter(float *image, uint32_t width, uint32_t height);

float *apply_edge_thinning(float *image, uint32_t width, uint32_t height);

float *apply_double_threshold(float *image, uint32_t width, uint32_t height);

float *apply_edge_histeresis(float *image, uint32_t width, uint32_t height);

// for sigma = 1.0
#define GAUSSIAN_KERNEL_SIZE 5
static const float gaussian_kernel[5][5] = {
        {1.0f / 273.0f, 4.0f / 273.0f,  7.0f / 273.0f,  4.0f / 273.0f,  1.0f / 273.0f},
        {4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f},
        {7.0f / 273.0f, 26.0f / 273.0f, 41.0f / 273.0f, 26.0f / 273.0f, 7.0f / 273.0f},
        {4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f},
        {1.0f / 273.0f, 4.0f / 273.0f,  7.0f / 273.0f,  4.0f / 273.0f,  1.0f / 273.0f}
};

static const float sobel_kernel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
};
static const float sobel_kernel_y[3][3] = {
        {1,  2,  1},
        {0,  0,  0},
        {-1, -2, -1}
};

int main(int argc, char **argv) {
    uint32_t error;
    uint8_t *image;
    uint32_t width, height;

    char inputImagePath[1024] = {0};
    char outputImagePath[1024] = {0};

    switch (argc) {
        case 1:
            strcpy(inputImagePath, "lenna.png");
            strcpy(outputImagePath, "/tmp/lena_out.png");
            break;
        case 2:
            assert(strstr(argv[1], ".png") != nullptr);
            strcpy(inputImagePath, argv[1]);
            strcpy(outputImagePath, "/tmp/lena_out.png");
            break;
        case 3:
            assert(strstr(argv[2], ".png") != nullptr);
            assert(strstr(argv[3], ".png") != nullptr);
            strcpy(inputImagePath, argv[1]);
            strcpy(outputImagePath, argv[2]);
            break;
        default:
            printf("Usage: %s [input_image_path] [output_image_path]\n", argv[0]);
            return 1;
    }

    printf("Input image path: %s\n", inputImagePath);
    printf("Output image path: %s\n", outputImagePath);
    printf("Using %d threads\n", omp_get_max_threads());

    error = lodepng_decode24_file(&image, &width, &height, inputImagePath);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    printf("The loaded image has dimensions %u x %u\n", width, height);

    // turn the image into a float array as it's easier to work with
    float *image_float = malloc(width * height * 3 * sizeof(float));
    for (uint32_t i = 0; i < width * height * 3; i++) {
        image_float[i] = (float) image[i] / 255.0f;
    }

    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);

    image_float = convert_to_grayscale(image_float, width, height);
    image_float = apply_gaussian_filter(image_float, width, height);
    image_float = apply_sobel_filter(image_float, width, height);
    image_float = apply_edge_thinning(image_float, width, height);
    image_float = apply_double_threshold(image_float, width, height);
    image_float = apply_edge_histeresis(image_float, width, height);

    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
#define TIME_IN_SECONDS(start, end) ((double) (end.tv_sec - start.tv_sec) + (double) (end.tv_nsec - start.tv_nsec) / 1000000000)

    printf("Time taken: %f seconds\n", TIME_IN_SECONDS(start, end));

    image = float_array_to_uint8_array(image_float, image, width, height);
    error = lodepng_encode_file(outputImagePath, image, width, height, LCT_GREY, 8);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

    free(image_float);
    free(image);
    return 0;
}

float *convert_to_grayscale(float *image, uint32_t width, uint32_t height) {
    float *new_image = malloc(width * height * sizeof(float));
#pragma omp parallel default(none) shared(image, new_image, width, height)
    for (uint32_t i = 0; i < width * height; i++) {
        new_image[i] = 0.2126f * image[i * 3] + 0.7152f * image[i * 3 + 1] + 0.0722f * image[i * 3 + 2];
    }

#ifdef WRITE_INTERMEDIATE_IMAGES
    write_intermediate_image("/tmp/test_after_grayscale.png", new_image, width, height);
#endif

    free(image);
    return new_image;
}

float *apply_gaussian_filter(float *image, uint32_t width, uint32_t height) {
    float *new_image = malloc(width * height * sizeof(float));
    int kernel_radius = GAUSSIAN_KERNEL_SIZE / 2;

#pragma omp parallel for default(none) shared(image, new_image, width, height, kernel_radius, gaussian_kernel) collapse(2)
    for (int y = 0; y < (int) height; y++) {
        for (int x = 0; x < (int) width; x++) {
            float new_pixel_value = 0.0f;

            for (int i = -kernel_radius; i <= kernel_radius; i++) {
                for (int j = -kernel_radius; j <= kernel_radius; j++) {
                    int current_x = x + j;
                    int current_y = y + i;
                    uint32_t pixel_index = calculate_index_with_wrap_around(current_x, current_y, width, height);
                    new_pixel_value += image[pixel_index] * gaussian_kernel[i + kernel_radius][j + kernel_radius];
                }
            }

            uint32_t pixel_index = y * width + x;
            new_image[pixel_index] = new_pixel_value;
        }
    }

#ifdef WRITE_INTERMEDIATE_IMAGES
    write_intermediate_image("/tmp/test_after_gaussian_filter.png", new_image, width, height);
#endif

    free(image);
    return new_image;
}

float *apply_sobel_filter(float *image, uint32_t width, uint32_t height) {
    float *new_image = malloc(width * height * 2 * sizeof(float));

#pragma omp parallel for default(none) shared(image, new_image, width, height, sobel_kernel_x, sobel_kernel_y) collapse(2)
    for (int y = 0; y < (int) height; y++) {
        for (int x = 0; x < (int) width; x++) {
            float sobel_x = 0.0f;
            float sobel_y = 0.0f;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int current_x = x + j;
                    int current_y = y + i;

                    uint32_t pixel_index = calculate_index_with_wrap_around(current_x, current_y, width, height);
                    sobel_x += image[pixel_index] * sobel_kernel_x[i + 1][j + 1];
                    sobel_y += image[pixel_index] * sobel_kernel_y[i + 1][j + 1];
                }
            }

            float magnitude = sqrtf(sobel_x * sobel_x + sobel_y * sobel_y);
            float orientation = atan2f(sobel_y, sobel_x);
            orientation = orientation * (180.0f / (float) M_PI); // convert to radians
            float orientation_rounded = roundf(orientation / 45.0f) * 45.0f;
            orientation_rounded = fmodf(orientation_rounded + 180.0f, 180.0f); // convert to 0-180
            new_image[y * width + x] = magnitude;
            new_image[width * height + y * width + x] = orientation_rounded;
        }
    }

#ifdef WRITE_INTERMEDIATE_IMAGES
    write_intermediate_image("/tmp/test_after_sobel_intensity_filter.png", new_image, width, height);
    write_intermediate_image("/tmp/test_after_sobel_orientation_filter.png", new_image + width * height, width, height);
#endif

    free(image);
    return new_image;
}

float *apply_edge_thinning(float *image, uint32_t width, uint32_t height) {
    float *new_image = malloc(width * height * sizeof(float));

#pragma omp parallel for default(none) shared(image, new_image, width, height) collapse(2)
    for (int y = 0; y < (int) height; y++) {
        for (int x = 0; x < (int) width; x++) {
            float q = 255.0f;
            float r = 255.0f;
            uint32_t angle_index = width * height + y * width + x;
            float angle = image[angle_index];

            // angle 0
            if (angle >= 0 && angle < 22.5) {
                q = image[calculate_index_with_wrap_around(x + 1, y, width, height)];
                r = image[calculate_index_with_wrap_around(x - 1, y, width, height)];
            } else if (angle >= 22.5 && angle < 67.5) { // angle 45
                q = image[calculate_index_with_wrap_around(x - 1, y + 1, width, height)];
                r = image[calculate_index_with_wrap_around(x + 1, y - 1, width, height)];
            } else if (angle >= 67.5 && angle < 112.5) { // angle 90
                q = image[calculate_index_with_wrap_around(x, y + 1, width, height)];
                r = image[calculate_index_with_wrap_around(x, y - 1, width, height)];
            } else if (angle >= 112.5 && angle < 157.5) { // angle 135
                q = image[calculate_index_with_wrap_around(x - 1, y - 1, width, height)];
                r = image[calculate_index_with_wrap_around(x + 1, y + 1, width, height)];
            }

            float intensity = image[y * width + x];
            if (intensity >= q && intensity >= r) {
                new_image[y * width + x] = intensity;
            } else {
                new_image[y * width + x] = 0.0f;
            }
        }
    }

#ifdef WRITE_INTERMEDIATE_IMAGES
    write_intermediate_image("/tmp/test_after_edge_thinning.png", new_image, width, height);
#endif

    free(image);
    return new_image;
}

float *apply_double_threshold(float *image, uint32_t width, uint32_t height) {
    float *new_image = malloc(width * height * sizeof(float));

    float high = FLT_MIN;
#pragma omp parallel for default(none) reduction(max:high) shared(image, width, height)
    for (uint32_t i = 0; i < width * height; i++) {
        if (image[i] > high) high = image[i];
    }

    float high_threshold = high * HIGH_THRESHOLD_RATIO;
    float low_threshold = high_threshold * LOW_THRESHOLD_RATIO;

#pragma omp parallel for default(none) shared(image, new_image, width, height, high_threshold, low_threshold)
    for (uint32_t i = 0; i < width * height; i++) {
        if (image[i] > high_threshold) {
            new_image[i] = STRONG_EDGE_PIXEL;
        } else if (image[i] > low_threshold) {
            new_image[i] = WEAK_EDGE_PIXEL;
        } else {
            new_image[i] = 0.0f;
        }
    }

#ifdef WRITE_INTERMEDIATE_IMAGES
    write_intermediate_image("/tmp/test_after_double_threshold.png", new_image, width, height);
#endif

    free(image);
    return new_image;
}

float *apply_edge_histeresis(float *image, uint32_t width, uint32_t height) {
    float *new_image = malloc(width * height * sizeof(float));

#pragma omp parallel for default(none) shared(image, new_image, width, height) collapse(2)
    for (int y = 0; y < (int) height; y++) {
        for (int x = 0; x < (int) width; x++) {
            if (image[y * width + x] == WEAK_EDGE_PIXEL) {
                if (image[calculate_index_with_wrap_around(x + 1, y, width, height)] == STRONG_EDGE_PIXEL ||
                    image[calculate_index_with_wrap_around(x - 1, y, width, height)] == STRONG_EDGE_PIXEL ||
                    image[calculate_index_with_wrap_around(x, y + 1, width, height)] == STRONG_EDGE_PIXEL ||
                    image[calculate_index_with_wrap_around(x, y - 1, width, height)] == STRONG_EDGE_PIXEL ||
                    image[calculate_index_with_wrap_around(x + 1, y + 1, width, height)] == STRONG_EDGE_PIXEL ||
                    image[calculate_index_with_wrap_around(x - 1, y - 1, width, height)] == STRONG_EDGE_PIXEL ||
                    image[calculate_index_with_wrap_around(x - 1, y + 1, width, height)] == STRONG_EDGE_PIXEL ||
                    image[calculate_index_with_wrap_around(x + 1, y - 1, width, height)] == STRONG_EDGE_PIXEL) {
                    new_image[y * width + x] = STRONG_EDGE_PIXEL;
                } else {
                    new_image[y * width + x] = 0.0f;
                }
            } else {
                new_image[y * width + x] = image[y * width + x];
            }
        }
    }

#ifdef WRITE_INTERMEDIATE_IMAGES
    write_intermediate_image("/tmp/test_after_edge_histeresis.png", new_image, width, height);
#endif

    free(image);
    return new_image;
}
