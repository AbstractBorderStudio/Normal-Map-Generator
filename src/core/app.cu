#include <app.cuh>

App::App(int _width, int _height, const char* _title)
{
	// Initialize GLFW and ImGui
	width = _width;
	height = _height;
	title = _title;

	// Initialize member variables
	window = nullptr;
	running = false;
	isToolActive = false;
	io = nullptr;
	
	// Initialize ImGui Docking and Window Flags
	dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
	window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
}

App::~App()
{
	// Cleanup resources
	cleanup();
}

bool App::Init()
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
	window = glfwCreateWindow(width, height, "Normal Map Generator", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return false;
	}
	else
	{
		std::cout << "GLFW initialized successfully" << std::endl;
	}
	
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, App::Framebuffer_size_callback);
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
	IMGUI_CHECKVERSION();
	ImGuiContext *imGuiContext = ImGui::CreateContext();
	io = &ImGui::GetIO();
	io->ConfigFlags |= ImGuiConfigFlags_DockingEnable; 		// Enable Docking
	io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;   // Enable Keyboard Controls
	io->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;    // Enable Gamepad Controls

	if (imGuiContext == nullptr)
	{
		std::cout << "Failed to create ImGui context" << std::endl;
		glfwTerminate();
		return false;
	}
	else
	{
		std::cout << "ImGui context created successfully" << std::endl;
	}

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);          	// Second param install_callback=true will install GLFW callbacks and chain to existing ones.
	ImGui_ImplOpenGL3_Init();

	// Variables
	return true;
}

void App::Run()
{
	// render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // process input
        ProcessInput(window);
        // -----

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        // -----
		
		// DockSpacev
        if (io->ConfigFlags & ImGuiConfigFlags_DockingEnable)
            ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), dockspace_flags);
        // -----
        
        // App interface
		ImGui::Begin("My First Tool", &isToolActive, window_flags);
        ImGui::End();
		ImGui::Begin("My First Tool1", &isToolActive, window_flags);
        ImGui::End();
		ImGui::Begin("My First Tool2", &isToolActive, window_flags);
        ImGui::End();
        // -----

		// opengl render
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        // ------

		// imgui rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        // ------

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        glfwSwapBuffers(window);
        glfwPollEvents();
        // ------
    }
}

void App::cleanup()
{
	// terminate imgui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void App::ProcessInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void App::Framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}