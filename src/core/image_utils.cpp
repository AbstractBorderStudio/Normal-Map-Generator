#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <image_utils.h>

unsigned char* core::ImageUtils::load_image(const std::string& filename, int* width, int* height, int* channels) {
	if (filename.empty()) {
		std::cerr << "Filename is empty." << std::endl;
		return nullptr;
	}
	return stbi_load(filename.c_str(), width, height, channels, 0);
}

int core::ImageUtils::save_image(const std::string& filename, const unsigned char* data, int width, int height, int channels) {
	if (channels < 0 || channels > 4) {
		std::cerr << "Unsupported number of channels: " << channels << std::endl; 
		return 0;
	}
	if (width <= 0 || height <= 0) {
		std::cerr << "Invalid image dimensions: " << width << "x" << height << std::endl;
		return 0;
	}
	if (!data) {
		std::cerr << "Image data is null." << std::endl;
		return 0;
	}
	std::cout << "Saving image to " << filename << " with dimensions " << width << "x" << height << " and " << channels << " channels." << std::endl;
	return stbi_write_png(filename.c_str(), width, height, channels, data, width * channels);
}