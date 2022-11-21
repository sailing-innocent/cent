#include "models.h"

#include <iostream>
#include <ing/app/gl_common.hpp>
#include <random>
#include <functional>

std::string _vertPath_canvas = "E:/assets/shaders/canvas/shader.vert";
std::string _fragPath_canvas = "E:/assets/shaders/canvas/shader.frag";
std::string _vertPath_scene = "E:/assets/shaders/glscene/shader.vert";
std::string _fragPath_scene = "E:/assets/shaders/glscene/shader.frag";

float disp(float d, float d_min, float d_max, float display_min = -0.8f, float display_max = 0.8f) {
    return (d - d_min) * (display_max - display_min)/(d_max - d_min) + display_min; 
}

void drawFn(ing::GLCommonApp& app, std::function<float(float)> fn, float x_min, float x_max, float y_min, float y_max)
{
    size_t N = 100;
    float start = x_min;
    float end = x_max;
    float gap = (end - start)/(N-1);
    std::vector<float> data;
    for (auto i = 0; i < N; i++) {
        data.push_back(start + i * gap);
    }

    float y_disp = disp(fn(data[0]), y_min, y_max);
    float x_disp = disp(data[0], x_min, x_max);
    ing::GLPoint startPoint(x_disp, y_disp);
    ing::GLPoint prevPoint = startPoint;
    for (auto i = 1; i < N; i++) {
        y_disp = disp(fn(data[i]), y_min, y_max);
        x_disp = disp(data[i], x_min, x_max);
        ing::GLPoint point(x_disp, y_disp);
        ing::GLLine line(prevPoint, point);
        // std::cout << point.vertices()[0] << "," << data[i] << std::endl;
        app.addLines(line);
        prevPoint = point;
    }
}

void drawAxis(ing::GLCommonApp& app, float x_min = -1.0, float x_max = 1.0, float y_min = -1.0, float y_max = 1.0)
{
    std::vector<float> blue{0.0f, 0.0f, 1.0f, 0.0f};
    ing::GLPoint x_left(x_min);
    x_left.setColor(blue);
    ing::GLPoint x_right(x_max);
    x_right.setColor(blue);
    ing::GLPoint y_top(0.0f, y_max);
    y_top.setColor(blue);
    ing::GLPoint y_buttom(0.0f, y_min);
    y_buttom.setColor(blue);
    ing::GLLine x_axis(x_left, x_right);
    ing::GLLine y_axis(y_buttom, y_top);
    app.addLines(x_axis);
    app.addLines(y_axis);
}

int main() 
{
    std::default_random_engine e;
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    ing::GLCommonApp app;
    auto f = [] (float x) { 
        const float PI = 3.14159265359;
        return std::sinf(2 * PI * x); 
    };
    drawFn(app, f, -1.0, 1.0, -1.0, 1.0);
    drawAxis(app);

    // sampling
    size_t sample_N = 10;
    float sample_x_min = 0.0f;
    float sample_x_max = 1.0f;
    std::vector<float> samples_x;
    std::vector<float> samples_y;
    for (auto i = 0; i < sample_N; i++) {
        float sample_x = sample_x_min + (sample_x_max - sample_x_min) * u(e);
        float sample_y = f(sample_x) + u(e) * 0.2 -0.1;
        samples_x.push_back(sample_x);
        samples_y.push_back(sample_y);

        float disp_x = disp(sample_x,-1.0, 1.0);
        float disp_y = disp(sample_y,-1.0, 1.0);
        ing::GLPoint sample_p(disp_x, disp_y);
        std::vector<float> yellow = { 1.0f, 1.0f, 0.0f, 1.0f };
        sample_p.setColor(yellow);
        // std::cout << sample_p.vertices()[0] << "," << sample_p.vertices()[1] << std::endl;
        app.addPoints(sample_p);
    }

    // modeling

    Polynomial poly(5);

    // std::cout << poly.forward(2) << std::endl;

    drawFn(app, poly, -1.0, 1.0, -1.0, 1.0);

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
