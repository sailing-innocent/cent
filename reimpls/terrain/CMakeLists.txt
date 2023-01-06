cmake_minimum_required(VERSION 3.18)
set(TARGET_NAME terrain)

file(GLOB_RECURSE HEADER_FILES "*.h")
file(GLOB_RECURSE SOURCE_FILES "*.cpp")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${HEADER_FILES} ${SOURCE_FILES})

set(GLAD_SOURCE "D:/source/glad/glad.c")
set(MAIN_INCLUDES "D:/includes")
set(GLFW_LIB_PATH "D:/libs/glfw")


list(APPEND TERRAIN_SOURCE
    ${GLAD_SOURCE}
    ${HEADER_FILES}
    ${SOURCE_FILES}
)

list(APPEND TERRAIN_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MAIN_INCLUDES}
)

list(APPEND TERRAIN_LIB_PATH
    ${GLFW_LIB_PATH}
)

list(APPEND TERRAIN_LIBS
    glfw3
)

add_executable(${TARGET_NAME} ${TERRAIN_SOURCE})
target_include_directories(${TARGET_NAME} PUBLIC ${TERRAIN_INCLUDES})
target_link_directories(${TARGET_NAME} PUBLIC ${TERRAIN_LIB_PATH})
target_link_libraries(${TARGET_NAME} PUBLIC ${TERRAIN_LIBS})