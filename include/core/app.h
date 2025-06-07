#pragma once

// Include OPENGL BACKEND
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Include STANDARD LIBRARIES
#include <iostream>

// Include IMGUI
#include <user_interface.h>

// Include app data
#include <app_data.h>
// Include image utilities
#include <image_utils.cuh>
// Include Normal Map Generator
#include <normal_map_generator.cuh>

namespace core
{
	class App
	{
		private:
			AppData appData;
			UserInterface userInterface;
		public:
			App(int _width = 800, int _height = 600, const char* _title = "CUDA OpenGL ImGui Application");
			~App();

			bool Init();
			void Run();
			void Cleanup();
			static void ProcessInput(GLFWwindow *window);
			static void Framebuffer_size_callback(GLFWwindow* window, int width, int height);
	};
}