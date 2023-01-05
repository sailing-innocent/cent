cmake_minimum_required(VERSION 3.18)

project(debugself)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIBSAIL_DIR "D:/repos/sail/build/Debug/")
set(LIBSAIL_INCLUDE "D:/repos/sail/include/")
set(LIBING_DIR "D:/repos/ing/build/Debug/")
set(LIBING_INCLUDE "D:/repos/ing/include/")

list(APPEND LIBS_DIR
    ${LIBSAIL_DIR}
    ${LIBING_DIR}
    "E:/libs/glfw"
    "E:/sdk/Vulkan/Lib"
)

list(APPEND INCLUDES
    ${LIBSAIL_INCLUDE}
    ${LIBING_INCLUDE}
    "E:/libs/eigen/"
    "E:/sdk/Vulkan/Include"
    "E:/includes"
    "includes"
)

list(APPEND LIBS
    "glfw3"
    "vulkan-1"
    "libsail" 
    "libing"
)

list(APPEND SOURCE
    "src/main.cc"
)

add_executable(debugself ${SOURCE})
target_link_directories(debugself PUBLIC ${LIBS_DIR})
target_link_libraries(debugself PUBLIC ${LIBS})
target_include_directories(debugself PUBLIC ${INCLUDES})
