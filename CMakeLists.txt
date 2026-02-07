cmake_minimum_required(VERSION 3.10)
project(avdet CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

#  CUDA and TensorRT 
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# Slamtec RPLidar SDK
set(RPLIDAR_SDK_DIR /home/sean/rplidar_sdk/sdk)
message(STATUS "RPLIDAR_SDK_DIR = ${RPLIDAR_SDK_DIR}")

# Grab ALL SDK sources recursively
file(GLOB_RECURSE RPLIDAR_SDK_SRC
  ${RPLIDAR_SDK_DIR}/src/*.cpp
  ${RPLIDAR_SDK_DIR}/src/*.c
)

# exclude mac windows and arduino, this was causing errors in build with lidar
list(FILTER RPLIDAR_SDK_SRC EXCLUDE REGEX "./arch/macOS/.")
list(FILTER RPLIDAR_SDK_SRC EXCLUDE REGEX ".*/arch/win32/*.")
list(FILTER RPLIDAR_SDK_SRC EXCLUDE REGEX "./arch/arduino/.")

list(LENGTH RPLIDAR_SDK_SRC RPLIDAR_SDK_SRC_LEN)

set(BUILD_BYTETRACK_TEST OFF CACHE BOOL "" FORCE)   # disable tests
add_subdirectory(external/ByteTrack-cpp)



add_executable(avdet
  src/main.cpp
  src/trt_yolo.cpp
  src/lidar_c1.cpp
  ${RPLIDAR_SDK_SRC} #compile sdk sources to this target 
)
# sdk public and internal headers 
target_include_directories(avdet PRIVATE
  ${RPLIDAR_SDK_DIR}/include
  ${RPLIDAR_SDK_DIR}/src
)
target_include_directories(avdet PRIVATE /usr/include/eigen3)

# Opencv , Trt and Cuda runtime , threads and realtime 
target_link_libraries(avdet PRIVATE
  ${OpenCV_LIBS}
  nvinfer nvonnxparser cudart 
  pthread rt
  bytetrack
)
