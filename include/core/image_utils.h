#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <iostream>
#include <string>

#include <stb_image.h>
#include <stb_image_write.h>

namespace core
{
	/**
	* @brief custom image structure to hold image data
	*/
	struct Image {
		unsigned char* data;
		int width;
		int height;
		int channels;
		Image() : data(nullptr), width(0), height(0), channels(0) {}
		Image(unsigned char* data, int width, int height, int channels)
			: data(data), width(width), height(height), channels(channels) {}
		~Image() {
			if (data) {
				stbi_image_free(data);
			}
		}
	};

	/**
	 * @brief Utility class for image loading and saving
	 */
	class ImageUtils {
		public:
			static unsigned char* load_image(const std::string& filename, int* width, int* height, int* channels);
			static int save_image(const std::string& filename, const unsigned char* data, int width, int height, int channels);
	};
}

#endif