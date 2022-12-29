#ifndef GLAD_H
#define GLAD_H
#include <glad/glad.h>
#endif // GLAD_H
#include <glfw/glfw3.h>
#include <glm/glm.hpp>

#include "engine/window.h"
#include <iostream>

int main() 
{
    int success;
    Window window(success, 1600, 900);
    if (!success) return -1;

    glm::vec3 fogColor(0.5,0.6,0.7);

    while (window.continueLoop())
    {
        glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
        glClearColor(fogColor[0], fogColor[1], fogColor[2], 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        window.swapBuffersAndPollEvents();
    }
    return 0;
}

