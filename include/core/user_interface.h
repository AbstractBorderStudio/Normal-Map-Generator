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
		public:
			UserInterface();

			bool TryInit(GLFWwindow* window);
			void Render(AppData *data);
			void Shutdown();
	};
}