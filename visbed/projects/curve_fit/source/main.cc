#include <iostream>
#include <ing/app/gl_common.hpp>
#include <cmath>

std::string _vertPath_canvas = "E:/assets/shaders/canvas/shader.vert";
std::string _fragPath_canvas = "E:/assets/shaders/canvas/shader.frag";
std::string _vertPath_scene = "E:/assets/shaders/glscene/shader.vert";
std::string _fragPath_scene = "E:/assets/shaders/glscene/shader.frag";

int main() 
{
    ing::GLCommonApp app;
    // generate split data
    size_t N = 100;

    float start = 0.0f;
    float end = 1.0f;
    float gap = (end - start)/(N-1);
    std::vector<float> data;
    for (auto i = 0; i < N; i++) {
        data.push_back(start + i * gap);
        // std::cout << start + i * gap << ",";
    }
    float y_range = 2.0f;
    float min_y = -0.8f;
    float max_y = 0.8f;
    float min_x = -0.8f;
    float max_x = 0.8f;
    float x_range = 1.0f;
    // generate points and lines
    auto f = [] (float x) { 
        const float PI = 3.14159265359;
        return std::sinf(2 * PI * x); 
    };
    ing::GLPoint startPoint(data[0]*(max_x - min_x)/x_range + min_x, (f(data[0]) + 1.0f )*(max_y-min_y)/y_range+min_y);
    ing::GLPoint prevPoint = startPoint;
    for (auto i = 1; i < N; i++) {
        ing::GLPoint point(data[i]*(max_x - min_x)/x_range + min_x, (f(data[i]) + 1.0f )*(max_y-min_y)/y_range + min_y);
        ing::GLLine line(prevPoint, point);
        // std::cout << point.vertices()[0] << "," << data[i] << std::endl;
        app.addLines(line);
        prevPoint = point;
    }
    app.init();
    size_t count;
    count = app.addShader(_vertPath_scene, _fragPath_scene);
    count = app.addShader(_vertPath_canvas, _fragPath_canvas);
    // std::cout << "Now we have " << count << " Shaders. " << std::endl;
    int i = 0;
    while (!app.shouldClose()) {
        app.tick(i);
        i++;
    }
    app.terminate();
    return 0;
}
