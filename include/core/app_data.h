#pragma once

// Include OPENGL BACKEND
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Include image utilities
#include <image_utils.h>

namespace core
{
	struct AppData
	{
		GLFWwindow* window;
		int width, height;
		const char* title;

		// input image reference
		Image inputImage;
		Image outputImage;

		AppData(int w, int h, const char* t)
			: window(nullptr), width(w), height(h), title(t), inputImage(), outputImage() {}
	};
}