#include <user_interface.h>

core::UserInterface::UserInterface() 
{
	isToolActive = false;
	io = nullptr;
	// Initialize ImGui Docking and Window Flags
	dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
	window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse;
	preview_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

	inputImagePath = "";
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
	if (ImGui::Button("Open File"))
	{
		if (TryOpenFileDialog(inputImagePath))
		{
			std::cout << "Selected file: " << inputImagePath << std::endl;
		}
	}
	
	ImGui::Text("Current Image Path: %s", inputImagePath.c_str());

	if (ImGui::Button("Load"))
		ImageUtils::TryLoadImage(inputImagePath, &data->inputImage);
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
		ImageUtils::SaveImage("../resources/output.png", &data->outputImage);
	}
    ImGui::End();
	// -----

	// Input preview and Output preview
	// -----
	ImGui::Begin("Input Preview", &isToolActive, preview_flags);
	if (ImGui::IsWindowHovered() && io->MouseWheel != 0.0f)
		inputZoom += ImGui::GetIO().MouseWheel * 0.1f;
	inputZoom = std::clamp(inputZoom, 0.1f, 20.0f);
	if (data->inputImage.IsValid())
	{
		auto img_size = ComputeDynamicImageSize(&data->inputImage, inputZoom);
		ImGui::Image(data->inputImage.textureID, img_size);
	}
	ImGui::End();

	// Output Preview
	ImGui::Begin("Output Preview", &isToolActive, preview_flags);
		if (ImGui::IsWindowHovered() && io->MouseWheel != 0.0f)
		outputZoom += ImGui::GetIO().MouseWheel * 0.1f;
	outputZoom = std::clamp(outputZoom, 0.01f, 20.0f);
	if (data->outputImage.IsValid())
	{
		auto img_size = ComputeDynamicImageSize(&data->outputImage, outputZoom);
		ImGui::Image(data->outputImage.textureID, img_size);
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

void EscapeWhitespace(std::string& path) {
	size_t pos = 0;
	while ((pos = path.find(' ', pos)) != std::string::npos) {
		path.replace(pos, 1, "\\ ");
		pos += 2; // Skip past the newly inserted \ and space
	}
}

bool core::UserInterface::TryOpenFileDialog(std::string &filePath) 
{
	char fileName[MAX_PATH] = "";

    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));

    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFilter = "PNG Files\0*.png\0";
    ofn.lpstrFile = fileName;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
    ofn.lpstrDefExt = "png";

    if (GetOpenFileNameA(&ofn)) {
        filePath = fileName;

        // Convert backslashes to forward slashes
        std::replace(filePath.begin(), filePath.end(), '\\', '/');
        //EscapeWhitespace(filePath);
        return true;
    }

    return false;
}

ImVec2 core::UserInterface::ComputeDynamicImageSize(Image *image, float zoom)
{
	float avail_height = ImGui::GetWindowHeight();
	float avail_width = ImGui::GetWindowWidth();
	float aspect = (float)image->width / (float)image->height;
	ImVec2 img_size = ImVec2(avail_height * aspect * zoom, avail_height * zoom);

	// Clamp image size to window width if needed
	if (img_size.x > avail_width) {
		img_size.x = avail_width;
		img_size.y = avail_width / aspect;
	}

	// Center image horizontally and vertically
	float x_offset = (avail_width - img_size.x) * 0.5f;
	float y_offset = (avail_height - img_size.y) * 0.5f;
	if (x_offset > 0 || y_offset > 0)
		ImGui::SetCursorPos(ImVec2(x_offset > 0 ? x_offset : 0, y_offset > 0 ? y_offset : 0));
	
	return img_size;
}