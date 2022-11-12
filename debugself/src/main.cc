#include <string>
#include <vector>
#include <iostream>
#include <sail/math.h>
#include <sail/ui/figure.h>
#include <ing/app/vk_canvas.hpp>
#include <cmath>

const std::string vertShaderPath = "E:/assets/shaders/canvas/vert.spv";
const std::string fragShaderPath = "E:/assets/shaders/canvas/frag.spv";

int drawHistogram(std::vector<float>& data) {
    if (data.size() == 0) { return -1; }
    auto _min = data[0];
    auto _max = data[0];
    for (auto item: data) {
        if (_min > item) { _min = item; }
        if (_max < item) { _max = item; }
    }
    sail::Histogram2D his(data, _max, _min);
    sail::VisNode vn;
    his.draw(vn);
    std::vector<float> vf;
    std::vector<uint16_t> iu;
    vn.genING(vf, iu);
    std::cout << "iu: ["; 
    for (auto i: iu) {
        std::cout << i << ",";
    }
    std::cout << "]" << std::endl;
    std::cout << "vf: ["; 
    for (auto i: vf) {
        std::cout << i << ",";
    }
    std::cout << "]" << std::endl;

    ing::CanvasApp app(vertShaderPath, fragShaderPath);

    if (!app.setVertex(vf, vf.size())) {
        std::cout << "Init Vertices Failed~" << std::endl;
    }
    if (!app.setIndex(iu, iu.size())) {
        std::cout << "Init Indices Failed" << std::endl;
    }

    
    try {
        app.init();
        app.run();
        app.terminate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return 0;
}
int drawTriangle() {
    sail::point p1(-0.5, 0.0, 0.0, 1.0);
    sail::point p2(0.5, 0.0, 0.0, 1.0);
    sail::point p3(0.0, 0.5, 0.0, 1.0);
    sail::Triangle tri(p1, p2, p3);
    sail::VTriangle vtri;
    tri.visualize(vtri);

    std::vector<float> vf;
    std::vector<uint16_t> iu;
    vtri.genING(vf, iu);

    ing::CanvasApp app(vertShaderPath, fragShaderPath);

    if (!app.setVertex(vf, vf.size())) {
        std::cout << "Init Vertices Failed~" << std::endl;
    }
    if (!app.setIndex(iu, iu.size())) {
        std::cout << "Init Indices Failed" << std::endl;
    }

    
    try {
        app.init();
        app.run();
        app.terminate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return 0;
}
int drawRect2D() {
    sail::point lb(-0.5f, -0.5f, 0.0f, 1.0f);
    float width = 1.0f;
    float height = 1.0f;
    sail::Rect2D rec(lb, width, height);
    sail::VisNode vn;
    rec.visualize(vn);
    std::vector<float> vf;
    std::vector<uint16_t> iu;
    vn.genING(vf, iu);

    std::cout << "["; 
    for (auto i: iu) {
        std::cout << i << ",";
    }
    std::cout << "]" << std::endl;
    // iu[5] = 3;
    ing::CanvasApp app(vertShaderPath, fragShaderPath);

    if (!app.setVertex(vf, vf.size())) {
        std::cout << "Init Vertices Failed~" << std::endl;
    }
    if (!app.setIndex(iu, iu.size())) {
        std::cout << "Init Indices Failed" << std::endl;
    }

    
    try {
        app.init();
        app.run();
        app.terminate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return 0;
}

int visualizeDistribution() {
    double _start = -3.0;
    double _end = 3.0;
    int segs = 18;
    double gap = (_end - _start) / segs;
    sail::GaussianDistribution1D gd;
    std::vector<float> data; // = {1.0f, 3.0f, 2.0f, 4.0f, 1.0f};
    for (auto i = 0; i < segs-1; i++) {
        data.push_back(static_cast<float>(gd(_start+i*gap, _start+(i+1)*gap)));
    }    

    int res = drawHistogram(data);
    return res;
}

int main() {
    int res = visualizeDistribution();
    return res;
}
