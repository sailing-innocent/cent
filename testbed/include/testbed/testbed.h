#pragma once
#ifndef TESTBED_H_
#define TESTBED_H_

#include <testbed/common.h>
#include <testbed/shared_queue.h>

struct GLFWwindow;

TESTBED_NAMESPACE_BEGIN

class TriangleOctree;
class Triangle;
class TriangleBvh;
class GLTexture;

// The common interface for Testbed
class Testbed {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit Testbed(ITestbedMode mode);
    virtual ~Testbed();
    // constructor with datapath
    // load_training_data
    // clear_training_data
    // distance_fn_t
    // normals_fun_t

    class SphereTracer {
    public:
        // SphereTracer() : m_hit_counter(1), m_alive_counter(1) {}
        // void init_rays_from_camera();
        // void init_rays_from_data
        // trace_bvh
        // trace
        // enlarge
        // rays_hit
        // rays_init
    private:
        // RaysSdfSoa
        
    };

    // class FiniteDiffereceNormalApproximator
    // Network Dims
    // render_volume
    // train_volume

    // void render_sdf()
    // render_nerf
    // void render_image(); // buffer, stream
    // void imgui();
    void init_window(int resw, int resh, bool hidden=false, bool second_window=false);
    void destroy_window();
    void draw_gui();
    bool frame();

    void render();

    // load_image
    // MeshState

    // struct Nerf

// ----------------------- GUI Relevant Method -------------------------- //
#ifdef ENABLE_GUI
    bool begin_frame();
#endif 
// ---------------------------------------------------------------------- //

// --------------------------- DATA MEMBER ----------------------------- //
    Eigen::Vector2i m_window_res = Eigen::Vector2i::Constant(0);
    ITestbedMode m_testbed_mode;
    bool m_render_window = true;
#ifdef ENABLE_GUI
    GLFWwindow* m_glfw_window = nullptr;
    bool m_gui_redraw = true;
    std::vector<std::shared_ptr<GLTexture>> m_render_texture;
#endif
// --------------------------------------------------------------------- //

};

TESTBED_NAMESPACE_END

#endif // TESTBED_H_