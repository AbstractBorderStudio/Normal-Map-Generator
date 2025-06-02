#include <user_interface.h>

core::UserInterface::UserInterface() 
{
	isToolActive = false;
	io = nullptr;
	// Initialize ImGui Docking and Window Flags
	dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
	window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse;
}

bool core::UserInterface::TryInit(GLFWwindow *window) 
{
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
	return true;
}

void core::UserInterface::Render(AppData *data) 
{
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
	ImGui::Begin("App", &isToolActive, window_flags);
	ImGui::SeparatorText("General");
	if (ImGui::Button("Load"))
		core::ImageUtils::TryLoadImage("../resources/source.png", &data->inputImage);
	if (ImGui::Button("Clear"))
		data->inputImage.Clear();
	if (ImGui::Button("Save"))
		core::ImageUtils::SaveImage("../resources/output.png", &data->inputImage);
    ImGui::End();
	if (data->inputImage.IsValid())
	{
		ImGui::Begin("Input Preview", &isToolActive, window_flags);
		ImGui::Image(data->inputImage.textureID, ImVec2(200, 200));
		ImGui::End();
	}
	if (data->outputImage.IsValid())
	{
		ImGui::Begin("Output Preview", &isToolActive, window_flags);
		ImGui::Image(data->outputImage.textureID, ImVec2(200, 200));
		ImGui::End();
	}
    // -----

	// imgui rendering
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void core::UserInterface::Shutdown() 
{
	// terminate imgui
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}