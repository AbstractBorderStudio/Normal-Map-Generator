#ifndef MAIN_H
#define MAIN_H

// Include STANDARD LIBRARIES
using namespace std;
#include <iostream>

// Include image utilities
#include <image_utils.h>

// Include OPENGL BACKEND
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Include DEAR IMGUI
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// Include Normal Map Generator
#include <normal_map_generator.cuh>

class App
{
	private:
		GLFWwindow* window;
		ImGuiIO* io;
		ImGuiDockNodeFlags dockspace_flags;
		ImGuiDockNodeFlags window_flags;
		int width, height;
		const char* title;
		bool running;
		bool isToolActive;
	public:
		// input image reference
		core::Image inputImage;
		core::Image outputlImage;

		App(int _width = 800, int _height = 600, const char* _title = "CUDA OpenGL ImGui Application");
		~App();

		bool Init();
		void Run();
		void cleanup();
		static void ProcessInput(GLFWwindow *window);
		static void Framebuffer_size_callback(GLFWwindow* window, int width, int height);
};

#endif