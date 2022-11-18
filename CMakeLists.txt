cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(cent VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF)

######################## BUILD TARGETS ###########################

set(TESTBED TRUE)


##################################################################


###################### DEPS DEFINITION ###########################

set(MAIN_INCLUDES "E:/includes")

set(IMGUI_DIR "E:/source/imgui/")
set(IMGUI_INCLUDE "${IMGUI_DIR}")
list(APPEND IMGUI_SOURCE
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
)

set(LIBSAIL_INCLUDE "D:/repos/sail/include")
set(LIBSAIL_DIR "D:/repos/sail/build/Debug")
set(LIBING_INCLUDE "D:/repos/ing/include")
set(LIBING_DIR "D:/repos/ing/build/Debug")

#################################################################

if (${TESTBED})
add_subdirectory(testbed)
endif()
