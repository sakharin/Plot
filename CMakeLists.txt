cmake_minimum_required(VERSION 2.8)
project(Plot)
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# The version number.
set (Plot_VERSION_MAJOR 0)
set (Plot_VERSION_MINOR 0)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")

include_directories("${PROJECT_SOURCE_DIR}")

# Geometry
add_library(Geometry STATIC Geometry.cpp)
target_link_libraries(Geometry ${OpenCV_LIBS})

add_executable(testGeometry testGeometry.cpp)
target_link_libraries(testGeometry Geometry)
target_link_libraries(testGeometry ${OpenCV_LIBS})
target_link_libraries(testGeometry ${GTEST_LIBRARIES} pthread)

add_library(EquiRecFeatureDetector STATIC EquiRecFeatureDetector.cpp)
target_link_libraries(EquiRecFeatureDetector Geometry)
target_link_libraries(EquiRecFeatureDetector  ${OpenCV_LIBS})

add_executable(testEquiRecFeatureDetector testEquiRecFeatureDetector.cpp)
target_link_libraries(testEquiRecFeatureDetector Geometry)
target_link_libraries(testEquiRecFeatureDetector EquiRecFeatureDetector)
target_link_libraries(testEquiRecFeatureDetector ${OpenCV_LIBS})
