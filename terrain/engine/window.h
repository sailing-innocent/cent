#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "utils/camera.h"
#include <iostream>

class Window
{
public:
    Window(int& success, unsigned int scrW = 1600, unsigned int scrH = 900, std::string name = "Terrain Engine OpenGL");
    ~Window();
    GLFWwindow* m_window;
    GLFWwindow* getWindow() const { return m_window; }

    void processInput(float frameTime); // input handler

    static unsigned int SCR_WIDTH;
    static unsigned int SCR_HEIGHT;

    void terminate() {
        glfwTerminate();
    }

    // wireframe
    bool continueLoop() {
        return !glfwWindowShouldClose(this->m_window);
    }

    void swapBuffersAndPollEvents() {
        glfwSwapBuffers(this->m_window);
        glfwPollEvents();
    }

    static Camera* m_camera;

private:
    int m_old_state, m_new_state;
    int gladLoader(); // set mouse input and load opengl functions

    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
    static void mouse_callback(GLFWwindow* window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

    // static bool keyBools[10];
    // static bool mouseCursorDisabled;
    // wireframe
    // firstMouse
    // lastX
    // lastY

    std::string m_name;
};


