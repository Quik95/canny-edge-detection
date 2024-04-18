cmake_minimum_required(VERSION 3.27)
project(edge_detection C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g3 -fopenmp -Wall -Wextra -Wno-unused -Wno-unknown-pragmas -fno-omit-frame-pointer")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(ENV{ASAN_OPTIONS} "abort_on_error=1:halt_on_error=1")
    set(ENV{UBSAN_OPTIONS} "abort_on_error=1:halt_on_error=1")
endif ()

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g3 -fsanitize=address -fsanitize=leak -fsanitize=undefined -fsanitize-trap=undefined")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native -flto")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendor/lodepng)
add_library(lodepng ${CMAKE_CURRENT_SOURCE_DIR}/vendor/lodepng/lodepng.c)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

add_executable(edge_detection main.c)
add_executable(opencl opencl.c)

target_link_libraries(opencl ${OpenCL_LIBRARIES})

target_link_libraries(edge_detection m)
target_link_libraries(edge_detection lodepng)
target_link_libraries(opencl lodepng)
