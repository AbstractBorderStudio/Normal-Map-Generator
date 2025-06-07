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
#include <shlobj.h>
#include <commdlg.h>
#include <string>

#include <chrono>

namespace core
{
	enum HardwareType
	{
		GPU = 0,
		CPU = 1
	};

	enum OptimizationType
	{
		BASE_ALGORITHM = 0,
		TILINIG = 1,
	};

	class UserInterface
	{
		private:
			// Private members
			ImGuiIO* io;
			ImGuiDockNodeFlags dockspaceFlags;
			ImGuiDockNodeFlags settingsWindowFlags;
			ImGuiDockNodeFlags previewWindowFlags;
			bool isToolActive;
			std::string inputImagePath;
			float inputZoom = 1.0f;
			float outputZoom = 1.0f;
			ImVec2 inputPanOffset = ImVec2(0.0f, 0.0f);
			ImVec2 outputPanOffset = ImVec2(0.0f, 0.0f);
			float normalMapStrength = 0.1f;
			float currentMapStrength = normalMapStrength;
			int hardwareType = 0; // Default to GPU
			int optimizationType = 0; // Default to base
			bool addPadding = false;
			bool useCornerPixels = false;
			int currentHardwareType = hardwareType;
			long timeNeeded = 0;

			// Private methods
			bool TryOpenFileDialog(std::string &filePath);
			ImVec2 ComputeDynamicImageSize(Image *image, float zoom, ImVec2 panOffset);
			void HandleInput(float *zoom, ImVec2 *panOffset);
			void RenderSettingsWindow(AppData *data);
			void RenderPreviewWindow(const char* previewName, Image *image, float *zoom, ImVec2 *panOffset);
			void LunchNormalMapGeneration(AppData *data);
		public:
			UserInterface();
			void HandleInput();
			bool TryInit(GLFWwindow* window);
			void Render(AppData *data);
			void Shutdown();
	};
}