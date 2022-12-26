cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(cent VERSION 0.1.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF)

######################## BUILD TARGETS ###########################

set(VISBED FALSE)
set(TERRAIN TRUE)

##################################################################
###################### DEPS DEFINITION ###########################

set(MAIN_INCLUDES "E:/includes")
set(MAIN_LIBS "E:/libs")

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

set(EIGEN_DIR "E:/libs/eigen/")
set(GLAD_SOURCE "E:/source/glad/glad.c")
#################################################################
if (${VISBED})
    add_subdirectory(visbed)
endif()

#################################################################
if (${TERRAIN})
    add_subdirectory(terrain)
endif()