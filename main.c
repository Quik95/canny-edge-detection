#include <stdio.h>
#include <malloc.h>
#include <stdint.h>
#include "vendor/lodepng/lodepng.h"

#define WRITE_INTERMEDIATE_IMAGES 1

static uint8_t *float_array_to_uint8_array(const float *in, uint8_t *out, uint32_t width, uint32_t height) {
    for (uint32_t i = 0; i < width * height * 3; i++) {
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
    uint8_t *image_uint8 = malloc(width * height * 3 * sizeof(uint8_t));
    image_uint8 = float_array_to_uint8_array(image, image_uint8, width, height);
    unsigned error = lodepng_encode24_file(filename, image_uint8, width, height);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    free(image_uint8);
}

#endif

float *apply_gaussian_filter(float *image, uint32_t width, uint32_t height);

// for sigma = 1.0
#define GAUSSIAN_KERNEL_SIZE 5
static const float gaussian_kernel[5][5] = {
        {1.0f / 273.0f, 4.0f / 273.0f,  7.0f / 273.0f,  4.0f / 273.0f,  1.0f / 273.0f},
        {4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f},
        {7.0f / 273.0f, 26.0f / 273.0f, 41.0f / 273.0f, 26.0f / 273.0f, 7.0f / 273.0f},
        {4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f},
        {1.0f / 273.0f, 4.0f / 273.0f,  7.0f / 273.0f,  4.0f / 273.0f,  1.0f / 273.0f}
};

int main() {
    uint32_t error;
    uint8_t *image;
    uint32_t width, height;

    error = lodepng_decode24_file(&image, &width, &height, "/tmp/test.png");
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

    // turn the image into a float array as it's easier to work with
    float *image_float = malloc(width * height * 3 * sizeof(float));
    for (uint32_t i = 0; i < width * height * 3; i++) {
        image_float[i] = (float) image[i] / 255.0f;
    }

    printf("The loaded image has dimensions %u x %u\n", width, height);
    image_float = apply_gaussian_filter(image_float, width, height);

    image = float_array_to_uint8_array(image_float, image, width, height);
    error = lodepng_encode24_file("/tmp/test_out.png", image, width, height);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

    free(image_float);
    free(image);
    return 0;
}

float *apply_gaussian_filter(float *image, uint32_t width, uint32_t height) {
    float *new_image = malloc(width * height * 3 * sizeof(float));
    int kernel_radius = GAUSSIAN_KERNEL_SIZE / 2;

    for (int y = 0; y < (int) height; y++) {
        for (int x = 0; x < (int) width; x++) {
            float new_pixel_value[3] = {0, 0, 0};

            for (int i = -kernel_radius; i <= kernel_radius; i++) {
                for (int j = -kernel_radius; j <= kernel_radius; j++) {
                    int current_x = x + j;
                    int current_y = y + i;

                    uint32_t pixel_index = calculate_index_with_wrap_around(current_x, current_y, width, height);
                    new_pixel_value[0] +=
                            image[pixel_index * 3] * gaussian_kernel[i + kernel_radius][j + kernel_radius];
                    new_pixel_value[1] +=
                            image[pixel_index * 3 + 1] * gaussian_kernel[i + kernel_radius][j + kernel_radius];
                    new_pixel_value[2] +=
                            image[pixel_index * 3 + 2] * gaussian_kernel[i + kernel_radius][j + kernel_radius];
                }
            }

            uint32_t pixel_index = y * width + x;
            new_image[pixel_index * 3] = new_pixel_value[0];
            new_image[pixel_index * 3 + 1] = new_pixel_value[1];
            new_image[pixel_index * 3 + 2] = new_pixel_value[2];
        }
    }

#ifdef WRITE_INTERMEDIATE_IMAGES
    write_intermediate_image("/tmp/test_after_gaussian_filter.png", new_image, width, height);
#endif

    free(image);
    return new_image;
}