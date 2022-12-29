#include "window.h"

unsigned int Window::SCR_WIDTH = 1600;
unsigned int Window::SCR_HEIGHT = 900;

// camera
// keybool
// wireframe
// firstmouse
// lastX
// lastY

// mouseCursorDisable

Window::Window(int& success, unsigned int scrW, unsigned int scrH, std::string name): m_name(name)
{
    Window::SCR_WIDTH = scrW;
    Window::SCR_HEIGHT = scrH;
    success = 1;

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_SAMPLES, 4);

    this->m_window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, m_name.c_str(), NULL, NULL);
    if (!this->m_window) {
        std::cout << "Failed to Create Window" << std::endl;
        glfwTerminate();
        success = 0;
        return;
    }

    glfwMakeContextCurrent(this->m_window);
	glfwSetFramebufferSizeCallback(this->m_window, &Window::framebuffer_size_callback);

    m_old_state = m_new_state = GLFW_RELEASE;

    success = gladLoader() && success;
	if (success) {
		std::cout << "GLFW window correctly initialized!" << std::endl;
	}
}

int Window::gladLoader() {

	// tell GLFW to capture our mouse
	glfwSetInputMode(this->m_window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return 0;
	}

	return 1;
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void Window::framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void Window::processInput(float frameTime) {
	if (glfwGetKey(this->m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(this->m_window, true);
}

Window::~Window()
{
	this->terminate();
}
