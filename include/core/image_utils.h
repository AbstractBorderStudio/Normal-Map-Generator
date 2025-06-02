#pragma once

#include <iostream>
#include <string>

#include <stb_image.h>
#include <stb_image_write.h>

#include <glad/glad.h>  
#include <GLFW/glfw3.h> 

namespace core
{
	/**
	* @brief custom image structure to hold image data
	*/
	struct Image {
		unsigned char* data;
		GLuint textureID;
		int width;
		int height;
		int channels;

		Image() : data(nullptr), textureID(0), width(0), height(0), channels(0) {}

		~Image() {
			Clear();
		}

		void Init(unsigned char* data, int width, int height, int channels)
		{
			this->data = data;
			this->width = width;
			this->height = height;
			this->channels = channels;
			this->textureID = 0;

			if (!IsValid()) {
				std::cerr << "Invalid image data provided." << std::endl;
				Clear();
			}

			// generate gltexture
			glGenTextures(1, &textureID);
			glBindTexture(GL_TEXTURE_2D, this->textureID);

			// Setup filtering parameters for display
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

			// Upload pixels into texture
			glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, this->width, this->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, this->data);
		}

		bool IsValid()
		{
			return data != nullptr && width > 0 && height > 0 && (channels > 0 && channels <= 4);
		}

		void Clear()
		{
			if (data) {
				stbi_image_free(data);
			}
			if (textureID != 0) {
				glDeleteTextures(1, &textureID);
			}
			data = nullptr;
			width = 0;
			height = 0;
			channels = 0;
		}
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