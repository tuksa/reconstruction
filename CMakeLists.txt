cmake_minimum_required(VERSION 3.10)
project(reconstruction LANGUAGES CXX C)   # add C for MPI support

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common visualization)
find_package(jsoncpp REQUIRED) 
find_package(Boost REQUIRED COMPONENTS filesystem thread)
find_package(Eigen3 REQUIRED)

# --- Include directories ---
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# --- Source files ---
# Add every .cpp your build needs here (e.g. src/BundleAdjuster.cpp)
set(SOURCES
    prac_main.cpp
    src/SfM.cpp
    # src/BundleAdjuster.cpp
)

add_executable(prac_main ${SOURCES})

target_link_libraries(prac_main
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    Boost::filesystem
    Boost::thread
)