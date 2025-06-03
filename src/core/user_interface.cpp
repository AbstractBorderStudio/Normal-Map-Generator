#include <user_interface.h>

core::UserInterface::UserInterface() 
{
	isToolActive = false;
	io = nullptr;
	// Initialize ImGui Docking and Window Flags
	dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
	window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse;
	preview_flags = ImGuiWindowFlags_AlwaysAutoResize;// | ImGuiWindowFlags_NoMove;
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
		ImageUtils::TryLoadImage("../resources/source.png", &data->inputImage);
	if (ImGui::Button("Process"))
	{
		data->normalMapGenerator.GenerateNormalMap(
			&data->inputImage, 
			&data->outputImage
		);
	}
	if (ImGui::Button("Clear"))
	{
		if (data->inputImage.IsValid())
		{
			data->inputImage.Clear();
		}
		if (data->outputImage.IsValid())
		{
			data->outputImage.Clear();
		}
	}
	if (ImGui::Button("Save"))
	{
		if (!data->outputImage.IsValid())
		{
			std::cout << "Output image is not valid!" << std::endl;
			return;
		}
		ImageUtils::SaveImage("../resources/output.png", &data->inputImage);
	}
    ImGui::End();
	// -----

	// Input preview and Output preview
	// -----
	ImGui::Begin("Input Preview", &isToolActive, preview_flags);
	if (data->inputImage.IsValid())
	{
		ImGui::Image(data->inputImage.textureID, ImVec2(400, 400));
	}
	ImGui::End();
	ImGui::Begin("Output Preview", &isToolActive, preview_flags);
	if (data->outputImage.IsValid())
	{
		ImGui::Image(data->outputImage.textureID, ImVec2(400, 400));
	}
	ImGui::End();
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