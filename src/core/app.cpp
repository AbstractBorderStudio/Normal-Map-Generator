#include <app.h>

core::App::App(int width, int height, const char* title)
	: appData(width, height, title), userInterface() {}

core::App::~App()
{
	// Cleanup resources
	Cleanup();
}

bool core::App::Init()
{
	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	#ifdef __APPLE__
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	#endif

	// glfw window creation
	// --------------------
	appData.window = glfwCreateWindow(appData.width, appData.height, "Normal Map Generator", NULL, NULL);
	if (appData.window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return false;
	}
	else
	{
		std::cout << "GLFW initialized successfully" << std::endl;
	}
	
	glfwMakeContextCurrent(appData.window);
	glfwSetFramebufferSizeCallback(appData.window, core::App::Framebuffer_size_callback);
	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return false;
	}
	else
	{
		std::cout << "GLAD initialized successfully" << std::endl;
	}

	// Setup Dear ImGui context
	// -----------------------------
	if (!userInterface.TryInit(appData.window))
	{
		std::cout << "Failed to initialize user interface" << std::endl;
		return false;
	}
	else
	{
		std::cout << "User interface initialized successfully" << std::endl;
	}

	// Variables
	return true;
}

void core::App::Run()
{
	// render loop
    // -----------
    while (!glfwWindowShouldClose(appData.window))
    {
        // process input
        ProcessInput(appData.window);
        // -----

		// opengl render
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        // ------

		userInterface.Render(&appData);
        // ------

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        glfwSwapBuffers(appData.window);
        glfwPollEvents();
        // ------
    }
}

void core::App::Cleanup()
{
	appData.Cleanup();

	// Cleanup user interface
	userInterface.Shutdown();	

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void core::App::ProcessInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void core::App::Framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}