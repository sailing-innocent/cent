cmake_minimum_required(VERSION 3.10)

project(paralib 
    VERSION 1.0
    DESCRIPTION "The Test Parallel library"
    LANGUAGES C CUDA CXX
)


if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)


if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-bigobj")
endif()

list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

set(CMAKE_CUDA_ARCHITECTURES 86)

list(APPEND PARALIB_SOURCE
    "paralib/paralib.cu"
)
list(APPEND PARALIB_INCLUDES
    "include"
)

list(APPEND EXAMPLE_SOURCE
    "example/main.cu"
)



add_library(paralib SHARED ${PARALIB_SOURCE})
target_include_directories(paralib PUBLIC ${PARALIB_INCLUDES})

# add_executable(example ${EXAMPLE_SOURCE})
# target_include_directories(example PUBLIC ${PARALIB_INCLUDES})
# target_link_directories(example PUBLIC ${EXAMPLE_LIB_DIRS})
# target_link_libraries(example PRIVATE paralib)
