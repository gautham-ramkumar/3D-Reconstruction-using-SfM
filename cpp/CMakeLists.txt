cmake_minimum_required(VERSION 3.16)
project(sfm_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc features2d calib3d)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

# Core pipeline library; later modules add their sources here.
add_library(sfm
  src/rng.cpp
  src/io.cpp
  src/geometry.cpp
  src/matching.cpp
  src/tracks.cpp
  src/camera.cpp
  src/mapping.cpp
  src/ba.cpp
  src/export.cpp
)
target_include_directories(sfm PUBLIC include)
target_link_libraries(sfm PUBLIC ${OpenCV_LIBS} Eigen3::Eigen Ceres::ceres)
target_compile_options(sfm PRIVATE -Wall -Wextra)

# Per-module checks, run against the real dataset.
add_executable(test_io tests/test_io.cpp)
target_link_libraries(test_io PRIVATE sfm)

add_executable(test_geometry tests/test_geometry.cpp)
target_link_libraries(test_geometry PRIVATE sfm)

add_executable(test_matching tests/test_matching.cpp)
target_link_libraries(test_matching PRIVATE sfm)

add_executable(test_tracks tests/test_tracks.cpp)
target_link_libraries(test_tracks PRIVATE sfm)

add_executable(test_mapping tests/test_mapping.cpp)
target_link_libraries(test_mapping PRIVATE sfm)

add_executable(test_ba tests/test_ba.cpp)
target_link_libraries(test_ba PRIVATE sfm)

add_executable(test_export tests/test_export.cpp)
target_link_libraries(test_export PRIVATE sfm)

# End-to-end pipeline (replaces SFM.ipynb).
add_executable(sfm_pipeline src/main.cpp)
target_link_libraries(sfm_pipeline PRIVATE sfm)
