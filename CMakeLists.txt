cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(unet_libtorch)

set(CMAKE_PREFIX_PATH /home/calyx/libtorch)

find_package(OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(
        unet_libtorch
        main.cpp
        model.cpp
        post_process.h
        ./MOT/Sort.h
        ./MOT/Sort.cpp
        ./MOT/Hungarian.cpp
        ./MOT/Hungarian.h
        ./MOT/KalmanTracker.h
        ./MOT/KalmanTracker.cpp
        ./MOT/MOT_detector.h
        ./MOT/MOT_detector.cpp
        )

target_link_libraries(unet_libtorch "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
#set_property(TARGET unet_libtorch PROPERTY CXX_STANDARD 14)