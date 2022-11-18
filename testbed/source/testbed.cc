/**
 * @file: testbed/testbed.cc
 * @author: sailing-innocent
 * @create: 2022-11-17
 * @desp: The Common Testbed
*/

#include <testbed/common.h>
#include <testbed/testbed.h>

#ifdef ENABLE_GUI

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
// imgui impl opengl3
// imguizmo
// gl
// stb_image
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#endif

#undef min
#undef max
#undef near
#undef far

TESTBED_NAMESPACE_BEGIN

Testbed::Testbed(ITestbedMode mode): m_testbed_mode(mode) 
{
    // check compute capability
    // init neural network
    // reset camera

    // set other constants
}

Testbed::~Testbed() {
    if (m_render_window) {
        destroy_window();
    }
}

void Testbed::init_window(int resw, int resh, bool hidden, bool second_window)
{
#ifndef ENABLE_GUI
    // throw error
#else
    m_window_res = { resw, resh };
    // glfw set error callback
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, hidden ? GLFW_FALSE : GLFW_TRUE);
    std::string title = "Testbed (";
    
    m_glfw_window = glfwCreateWindow(m_window_res.x(), m_window_res.y(), title.c_str(), NULL, NULL);
    if (m_glfw_window == NULL) {
		throw std::runtime_error{"GLFW window could not be created."};
	}
	glfwMakeContextCurrent(m_glfw_window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        throw std::runtime_error{"GLAD could not be initialized."};
    }
    glfwSwapInterval(0); // disable vsync
    glfwSetWindowUserPointer(m_glfw_window, this);
    // set drop callback
    // set key callback
    // set cursor callback
    // set mouse button callback
    // set scroll callback
    // set frame buffer callback
    // float xscale, yscale;
    // glfwGetWindowContextScale(m_glfw_window, &xscale, &yscale);

    // IMGUI init
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    // imgui io
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(m_glfw_window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    // font configure

    // render_textures
    // render_surface

    m_render_window = true;

    // m_second_window
#endif // ENABLE_GUI
}

void Testbed::destroy_window()
{
#ifndef ENABLE_GUI
    throw std::runtime_error("destroy_window failed: TESTBED was built without GUI");
#else
    if (!m_render_window) {
        throw std::runtime_error("Window must be init to be destroyed");
    }
    // clear surface
    // clear texture
    // clear pip surface
    // clear pip texture
    // clear dlss

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(m_glfw_window);
    glfwTerminate();
    m_glfw_window = nullptr;
    m_render_window = false;
#endif
}

bool Testbed::frame()
{
#ifdef ENABLE_GUI
    if (m_render_window) {
        if (!begin_frame()) {
            return false;
        }
    }
#endif

    redraw_gui_next_frame();
    // clear the exsiting tasks and prepare data
    try {

    } catch (SharedQueueEmptyException&) {}

    // train_and_render --> prepare the images in m_render_textures
    // if mode== Sdf
#ifdef ENABLE_GUI
    if (m_render_window) {
        if (m_gui_redraw) {
            // gather_histograms()
            draw_gui();
            m_gui_redraw = false;
        }

        ImGui::EndFrame();
    }

#endif 
    return true;
}

void Testbed::render()
{
#ifdef ENABLE_GUI
    // m_render_textures.front()->blit_from_cuda_mapping();
    
#endif // ENABLE_GUI
}

#ifdef ENABLE_GUI

bool Testbed::begin_frame()
{
    if (glfwWindowShouldClose(m_glfw_window)) {
        destroy_window();
        return false;
    }

    {

    }
    glfwPollEvents();
    glfwGetFramebufferSize(m_glfw_window, &m_window_res.x(), &m_window_res.y());
    
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // events
    // toggle
    return true;
}

void Testbed::draw_gui()
{
    // make sure all the cuda code finishes 
    // if render_textures not empty, m_second_window.draw(->texture)

    glfwMakeContextCurrent(m_glfw_window);
    int display_w, display_h;
    glfwGetFramebufferSize(m_glfw_window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.f, 0.f, 0.f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ImDrawList* list = ImGui::GetBackgroundDrawList();
    // list->AddImageQuad(()
    // ImDrawList = ImGui::GetBackgroundDrawList();
    // list->AddCallabke()
    // list->AddImageQuad((ImTextureID)(size_t)m_render_texture.front()->texture(), 00,w0,wh,0h, 00, 10, 11, 01)
    list->AddText(ImVec2(4.f, 4.f), 0xffffffff, "Ground Truth");
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(m_glfw_window);
    glFinish();
}

#endif

TESTBED_NAMESPACE_END

