#pragma once
#ifndef TESTBED_H_
#define TESTBED_H_

#include <testbed/common.h>
#include <testbed/shared_queue.h>

TESTBED_NAMESPACE_BEGIN


// The common interface for Testbed
class Testbed {
public:
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit Testbed(ITestBedMode mode);
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

};

TESTBED_NAMESPACE_END

#endif // TESTBED_H_