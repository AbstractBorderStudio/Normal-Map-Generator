#pragma once

#include <iostream>
#include <string>

#include <stb_image.h>
#include <stb_image_write.h>
#include <stb_image_resize.h>

#include <glad/glad.h>  
#include <GLFW/glfw3.h> 

#include <vector>

namespace core
{
	/**
	* @brief custom image structure to hold image data
	*/
	struct Image {
		std::vector<unsigned char> data;
		GLuint textureID;
		std::string imagePath;
		int width;
		int height;
		int channels;

		Image() : data(std::vector<unsigned char>()), textureID(0), width(0), height(0), channels(0) {}

		~Image() {
			Clear();
		}
		void Init(std::string imagePath, unsigned char* data, int width, int height, int channels);
		bool IsValid();
		void UpdateTexture();
		void Clear();
	};

	/**
	 * @brief Utility class for image loading and saving
	 */
	class ImageUtils {
		public:
			static bool TryLoadImage(const std::string& filename, Image *image);
			static int SaveImage(const std::string& filename, Image *image);
	};
}