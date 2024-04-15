#include <stdio.h>
#include <malloc.h>
#include <unistd.h>
#include "vendor/lodepng/lodepng.h"

int main() {
    unsigned error;
    unsigned char *image;
    unsigned width, height;

    error = lodepng_decode24_file(&image, &width, &height, "/tmp/test.png");
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

    printf("The loaded image has dimensions %u x %u\n", width, height);

    sleep(20);
    free(image);
    return 0;
}