#pragma once

#include <iostream>
#include <string>

#include <stb_image.h>
#include <stb_image_write.h>

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

		void Init(std::string imagePath, unsigned char* data, int width, int height, int channels)
		{
			this->imagePath = imagePath;
			this->data.assign(data, data + (width * height * channels));
			this->width = width;
			this->height = height;
			this->channels = channels;
			this->textureID = -1;

			// generate gltexture
			glGenTextures(1, &textureID);
			glBindTexture(GL_TEXTURE_2D, this->textureID);

			// Setup filtering parameters for display
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

			// Upload pixels into texture
			glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, this->width, this->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, this->data.data());
		}

		bool IsValid()
		{
			if (data.empty() || width <= 0 || height <= 0 || channels < 0 || channels > 4) {
				return false;
			}
			return true;
		}

		void UpdateTexture()
		{
			if (textureID != 0 && data.empty())
			{
				glBindTexture(GL_TEXTURE_2D, textureID);
				glTexSubImage2D(
					GL_TEXTURE_2D,
					0, // mipmap level
					0, 0, // xoffset, yoffset
					width,
					height,
					channels == 4 ? GL_RGBA : GL_RGB, // choose format based on channels
					GL_UNSIGNED_BYTE,
					data.data()
				);
			}
		}

		void Clear()
		{
			if (data.empty() != 0) {
				stbi_image_free(static_cast<unsigned char*>(data.data()));
				data.clear();
			}
			if (textureID != 0) {
				glDeleteTextures(1, &textureID);
			}
			width = 0;
			height = 0;
			channels = 0;
			imagePath.clear();
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