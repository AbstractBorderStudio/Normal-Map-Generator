#include <user_interface.h>

core::UserInterface::UserInterface() 
{
	isToolActive = false;
	io = nullptr;
	// Initialize ImGui Docking and Window Flags
	dockspaceFlags = ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_AutoHideTabBar;
	settingsWindowFlags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoMove;
	previewWindowFlags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoMove;

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
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), dockspaceFlags);
    
    // Render Settings Window
	RenderSettingsWindow(data);

	// Render Input preview
	RenderPreviewWindow("Input preview", &data->inputImage, &inputZoom, &inputPanOffset);

	// Render Output Preview
	RenderPreviewWindow("Output preview" , &data->outputImage, &outputZoom, &outputPanOffset);

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
	ImVec2 img_size = ImVec2((float)image->width * aspect * zoom, (float)image->height * aspect * zoom);

	// Clamp image size to window width if needed
	if (img_size.x > avail_width) {
		img_size.x = avail_width;
		img_size.y = avail_width / aspect;
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
		*zoom += ImGui::GetIO().MouseWheel * 0.05f;
	*zoom = std::clamp(*zoom, 0.05f, 1.0f);

	// Handle pan (drag with right mouse button)
	if (ImGui::IsWindowHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
		ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
		panOffset->x += drag_delta.x;
		panOffset->y += drag_delta.y;
		ImGui::ResetMouseDragDelta(ImGuiMouseButton_Right);
	}
}

void core::UserInterface::RenderSettingsWindow(AppData *data)
{
	ImGui::Begin("Normal Map Generator Settings", &isToolActive, settingsWindowFlags);
	if (ImGui::Button("Open File"))
	{
		if (TryOpenFileDialog(inputImagePath))
		{
			std::cout << "Selected file: " << inputImagePath << std::endl;
			if (ImageUtils::TryLoadImage(inputImagePath, &data->inputImage))
			{
				// if image is loaded successfully, update the texture
				data->normalMapGenerator.GenerateNormalMap(
					&data->inputImage, 
					&data->outputImage,
					currentMapStrength
				);
			}
			
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
	if (ImGui::Button("Reset Pan and Zoom"))
	{
		inputZoom = 1.0f;
		outputZoom = 1.0f;
		inputPanOffset = ImVec2(0.0f, 0.0f);
		outputPanOffset = ImVec2(0.0f, 0.0f);
	}
	ImGui::Separator();
	ImGui::Text("Normal Map Strength:");
	ImGui::SliderFloat("Strength", &normalMapStrength, 0.0f, 1.0f, "%.8f");
	if (normalMapStrength != currentMapStrength)
	{
		currentMapStrength = normalMapStrength;
		data->normalMapGenerator.GenerateNormalMap(
			&data->inputImage, 
			&data->outputImage,
			currentMapStrength
		);
	}

	ImGui::Text("Input Image Size: %dx%d", data->inputImage.width, data->inputImage.height);
	ImGui::Text("Output Image Size: %dx%d", data->outputImage.width, data->outputImage.height);
    ImGui::End();
}

void core::UserInterface::RenderPreviewWindow(const char* previewName, Image *image, float *zoom, ImVec2 *panOffset) 
{
	ImGui::Begin(previewName, &isToolActive, previewWindowFlags);
	HandleInput(zoom, panOffset);
	if (image->IsValid())
	{
		auto img_size = ComputeDynamicImageSize(image, *zoom, *panOffset);
		ImGui::Image(image->textureID, img_size);
	}
	ImGui::End();
}