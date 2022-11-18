/**
 * @file: testbed/testbed.cc
 * @author: sailing-innocent
 * @create: 2022-11-17
 * @desp: The Common Testbed
*/

#include <testbed/common.h>
#include <testbed/testbed.h>

#ifdef ENABLE_GUI

// imgui
// gl
// stb_image

#endif

#undef min
#undef max
#undef near
#undef far

TESTBED_NAMESPACE_BEGIN

Testbed::~Testbed() {}

void Testbed::init_window(int resw, int resh, bool hidden, bool second_window)
{
#ifndef ENABLE_GUI
    // throw error
#endif

    // create a window for m_glfw_window
    // m_render_window = true
}

void Testbed::destroy_window()
{
#ifndef ENABLE_GUI
    // throw error
#endif
    // check if m_render_window
    // clear surface
    // clear texture
    // clear pip surface
    // clear pip texture
    // clear dlss

    // ImGui_ImplOpenGL3_Shutdown()
    // ImGui_ImplGlfw_Shutdown();
    // ImGui::DestroyContext();
    // glfwDestroyWindow(m_glfw_window)
    // glfwTerminate();
    // m_glfw_window = nullptr
    // m_render_window = false
}

bool Testbed::frame()
{
#ifdef ENABLE_GUI
    // begin_frame_and_handle_user_input
#endif

    // clear the exsiting tasks and prepare data
    try {
        while (true) {
            // (*m_task_queue.tryPop())();
        }
    } catch (SharedQueueEmptyException&) {}

    // train_and_render
    // if mode== Sdf
#ifdef ENABLE_GUI
    // if m_render_window
        // if g_gui_redraw
        // draw_gui
    // ImGui::EndFrame();
#endif 
    return true;
}

void Testbed::render()
{

}

void Testbed::draw_gui()
{
    // glfwMakeContextCurrent
    // get frame buffer size
    // viewport

    // ImDrawList = ImGui::GetBackgroundDrawList();
    // list->AddCallabke()
    // list->AddImageQuad((ImTextureID)(size_t)m_render_texture.front()->texture(), 00,w0,wh,0h, 00, 10, 11, 01)
    // That's all...
}

TESTBED_NAMESPACE_END

