cmake_minimum_required(VERSION 2.8)
project(Plot)
set(CMAKE_BUILD_TYPE Release)
find_package( OpenCV REQUIRED )

# The version number.
set (Plot_VERSION_MAJOR 0)
set (Plot_VERSION_MINOR 0)

include_directories("${PROJECT_SOURCE_DIR}")

# Geometry
add_library(Geometry STATIC Geometry.cpp)
target_link_libraries(Geometry ${OpenCV_LIBS})

add_executable(testGeometry testGeometry.cpp)
target_link_libraries(testGeometry Geometry)
target_link_libraries(testGeometry ${OpenCV_LIBS})

add_library(EquiRecFeatureDetector STATIC EquiRecFeatureDetector.cpp)
target_link_libraries(EquiRecFeatureDetector Geometry)
target_link_libraries(EquiRecFeatureDetector  ${OpenCV_LIBS})

add_executable(testEquiRecFeatureDetector testEquiRecFeatureDetector.cpp)
target_link_libraries(testEquiRecFeatureDetector Geometry)
target_link_libraries(testEquiRecFeatureDetector EquiRecFeatureDetector)
target_link_libraries(testEquiRecFeatureDetector ${OpenCV_LIBS})
