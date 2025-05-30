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

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

#endif