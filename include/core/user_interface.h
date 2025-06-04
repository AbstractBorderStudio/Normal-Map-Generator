#pragma once

#include <iostream>
#include <app_data.h>

// Include OPENGL BACKEND
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Include DEAR IMGUI
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <windows.h>
#include <commdlg.h>
#include <string>

namespace core
{
	class UserInterface
	{
		private:
			ImGuiIO* io;
			ImGuiDockNodeFlags dockspace_flags;
			ImGuiDockNodeFlags window_flags;
			ImGuiDockNodeFlags preview_flags;
			bool isToolActive;
			std::string inputImagePath;
			float inputZoom = 1.0f;
			float outputZoom = 1.0f;

			bool TryOpenFileDialog(std::string &filePath);
			ImVec2 ComputeDynamicImageSize(Image *image, float zoom);
		public:
			UserInterface();

			void HandleInput();
			bool TryInit(GLFWwindow* window);
			void Render(AppData *data);
			void Shutdown();
	};
}