cmake_minimum_required(VERSION 3.10)
project(avdet CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

add_executable(avdet src/main.cpp src/trt_yolo.cpp)
target_link_libraries(avdet ${OpenCV_LIBS} nvinfer nvonnxparser cudart)
