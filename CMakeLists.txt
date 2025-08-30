cmake_minimum_required(VERSION 3.10)
project(SfMProject)

set(CMAKE_CXX_STANDARD 17)

# Find packages
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common visualization)

# Include directories
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

# Create executable
add_executable(main 
    main.cpp 
    src/SfM.cpp
)

# Link libraries
target_link_libraries(main 
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES}
)

target_compile_definitions(main PRIVATE ${PCL_DEFINITIONS})