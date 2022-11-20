cmake_minimum_required(VERSION 3.18)

set(TARGET_NAME curve_fit)

file(GLOB_RECURSE HEADER_FILES "*.h")
file(GLOB_RECURSE SOURCE_FILES "*.cc")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${HEADER_FILES} ${SOURCE_FILES})

add_executable(${TARGET_NAME} ${HEADER_FILES} ${SOURCE_FILES})

list(APPEND CURVE_FIT_DEPS_DIR
    ${LIBSAIL_DIR}
    ${LIBING_DIR}
    "E:/libs/glfw"
)

list(APPEND CURVE_FIT_DEPS_LIB
    libsail
    libing
    glfw3
)

list(APPEND CURVE_FIT_INCLUDES
    ${LIBSAIL_INCLUDE}
    ${LIBING_INCLUDE}
    "E:/includes"
)

target_include_directories(${TARGET_NAME} PUBLIC ${CURVE_FIT_INCLUDES})
target_link_directories(${TARGET_NAME} PUBLIC ${CURVE_FIT_DEPS_DIR})
target_link_libraries(${TARGET_NAME} PUBLIC ${CURVE_FIT_DEPS_LIB})
