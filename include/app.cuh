#ifndef MAIN_H
#define MAIN_H

// Include STANDARD LIBRARIES
using namespace std; 
#include <iostream>

// INCLUDE CUDA LIBRARY
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

// INCLUDE MATH LIBRARY
#include <linmath.h>

// Include OPENGL BACKEND
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Include DEAR IMGUI
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

class App
{
	private:
		GLFWwindow* window;
		ImGuiIO* io;
		ImGuiDockNodeFlags dockspace_flags;
		int width, height;
		const char* title;
		bool running;
		bool isToolActive;
	public:
		App(int _width = 800, int _height = 600, const char* _title = "CUDA OpenGL ImGui Application");
		~App();

		bool Init();
		void Run();
		void cleanup();
		static void ProcessInput(GLFWwindow *window);
		static void Framebuffer_size_callback(GLFWwindow* window, int width, int height);
};

#endif