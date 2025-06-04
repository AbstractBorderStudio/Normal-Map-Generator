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
	ImGui::SeparatorText("Settings");
	if (ImGui::Button("Open File"))
	{
		if (TryOpenFileDialog(inputImagePath))
		{
			std::cout << "Selected file: " << inputImagePath << std::endl;
			ImageUtils::TryLoadImage(inputImagePath, &data->inputImage);
		}
	}
	if (!inputImagePath.empty())
	{
		ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(100, 100, 100, 255)); // light gray
		ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
		ImGui::BeginChild("##InputPathBox", ImVec2(0, ImGui::GetTextLineHeight() * 3), false, ImGuiWindowFlags_NoScrollbar);

		// Add padding: 8px left/right, 4px top/bottom
		ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 8.0f);
		ImGui::Dummy(ImVec2(0, 4.0f)); // top padding
		ImGui::TextWrapped("%s", inputImagePath.c_str());
		ImGui::Dummy(ImVec2(0, 4.0f)); // bottom padding

		ImGui::EndChild();
		ImGui::PopStyleVar();
		ImGui::PopStyleColor();
	}
	ImGui::SeparatorText("Settings/Actions");
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
		if (data->outputImage.IsValid())
		{
			std::string savePath;
			if (TryOpenFileDialog(savePath))
			{
				if (!savePath.empty())
				{
					ImageUtils::SaveImage(savePath, &data->outputImage);
					std::cout << "Saving to: " << savePath << std::endl;
				}
			}
		}
		else
		{
			std::cout << "No output image to save." << std::endl;
		}
	}
    ImGui::End();
	// -----

	// Input preview and Output preview
	// -----
	ImGui::Begin("Input Preview", &isToolActive, preview_flags);
	HandleInput(&inputZoom, &inputPanOffset);
	if (data->inputImage.IsValid())
	{
		auto img_size = ComputeDynamicImageSize(&data->inputImage, inputZoom, inputPanOffset);
		ImGui::Image(data->inputImage.textureID, img_size);
	}
	ImGui::End();

	// Output Preview
	ImGui::Begin("Output Preview", &isToolActive, preview_flags);
	HandleInput(&outputZoom, &outputPanOffset);
	if (data->outputImage.IsValid())
	{
		auto img_size = ComputeDynamicImageSize(&data->outputImage, outputZoom, outputPanOffset);
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
    ofn.Flags = OFN_PATHMUSTEXIST;
    ofn.lpstrDefExt = "png";

    if (GetOpenFileNameA(&ofn)) {
        filePath = fileName;

        // Convert backslashes to forward slashes
        std::replace(filePath.begin(), filePath.end(), '\\', '/');
        return true;
    }

    return false;
}

ImVec2 core::UserInterface::ComputeDynamicImageSize(Image *image, float zoom, ImVec2 panOffset)
{
	float avail_height = ImGui::GetWindowHeight();
	float avail_width = ImGui::GetWindowWidth();
	float aspect = (float)image->width / (float)image->height;
	ImVec2 img_size = ImVec2((float)image->width * zoom, (float)image->height * zoom);

	// Clamp image size to window width if needed
	if (img_size.x > avail_width) {
		img_size.x = avail_width;
		img_size.y = avail_width;
	}

	// Center image horizontally and vertically
	float x_offset = (avail_width - img_size.x) * 0.5f + panOffset.x;
	float y_offset = (avail_height - img_size.y) * 0.5f + panOffset.y;;
	if (x_offset > 0 || y_offset > 0)
		ImGui::SetCursorPos(ImVec2(x_offset > 0 ? x_offset : 0, y_offset > 0 ? y_offset : 0));
	
	return img_size;
}

void core::UserInterface::HandleInput(float *zoom, ImVec2 *panOffset) 
{
	// Handle zoom
	if (ImGui::IsWindowHovered() && io->MouseWheel != 0.0f)
		*zoom += ImGui::GetIO().MouseWheel * 0.1f;
	*zoom = std::clamp(*zoom, 0.1f, 20.0f);

	// Handle pan (drag with right mouse button)
	if (ImGui::IsWindowHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
		ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
		panOffset->x += drag_delta.x;
		panOffset->y += drag_delta.y;
		ImGui::ResetMouseDragDelta(ImGuiMouseButton_Right);
	}
}