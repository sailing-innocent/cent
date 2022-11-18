set(TARGET_NAME testbed)

file(GLOB TESTBED_HEADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB TESTBED_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cc)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${TESTBED_HEADERS} ${TESTBED_SOURCES})

list(APPEND TESTBED_INCLUDES
    "${MAIN_INCLUDES}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${EIGEN_DIR}"
)

set(ENABLE_GUI TRUE)
if(ENABLE_GUI)
    list(APPEND TESTBED_INCLUDES
        ${IMGUI_INCLUDE}
    )
    list(APPEND IMGUI_SOURCE
        "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
        "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
    )
    list(APPEND TESTBED_SOURCES
        ${IMGUI_SOURCE}
        ${GLAD_SOURCE}
    )
    list(APPEND TESTBED_LIBS
        glfw/glfw3
    )
    list(APPEND TESTBED_LIB_DIR
        "E:/libs"
    )
    list(APPEND TESTBED_DIFINITIONS
        "-DENABLE_GUI"
    )
endif()

add_executable(${TARGET_NAME} ${TESTBED_HEADERS} ${TESTBED_SOURCES})
target_include_directories(${TARGET_NAME} PUBLIC ${TESTBED_INCLUDES})
target_compile_definitions(${TARGET_NAME} PUBLIC ${TESTBED_DIFINITIONS})
target_link_libraries(${TARGET_NAME} PUBLIC ${TESTBED_LIBS})
target_link_directories(${TARGET_NAME} PUBLIC ${TESTBED_LIB_DIR})