cmake_minimum_required(VERSION 2.8)
project(Plot)
set(CMAKE_BUILD_TYPE Release)
find_package( OpenCV REQUIRED )

# The version number.
set (Plot_VERSION_MAJOR 0)
set (Plot_VERSION_MINOR 0)

# Geometry
add_library(Geometry STATIC Geometry.cpp)
target_link_libraries(Geometry ${OpenCV_LIBS})
include_directories("${PROJECT_SOURCE_DIR}")

add_executable(testGeometry testGeometry.cpp)
target_link_libraries(testGeometry Geometry)
target_link_libraries(testGeometry ${OpenCV_LIBS})
