#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <image_utils.h>

bool core::ImageUtils::TryLoadImage(const std::string& filename, Image *image) {
	// Check if the filename is valid
	if (filename.empty()) {
		std::cerr << "Filename is empty." << std::endl;
		return false;
	}

	if (image == nullptr) {
		std::cerr << "Image pointer is null." << std::endl;
		image = new Image();
	}

	// Reset the image structure
	if (image->IsValid())
	{
		std::cerr << "Image is already valid, clearing previous data." << std::endl;
		image->Clear();
	}

	// load the image data
	int width, height, channels;
	unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
	if (!data) {
		std::cerr << "Failed to load image: " << filename << " - " << stbi_failure_reason() << std::endl;
		return false;
	}
	
	// Validate image dimensions and channels
	if (width <= 0 || height <= 0) {
		std::cerr << "Invalid image dimensions: " << width << "x" << height << std::endl;
		stbi_image_free(data);
		return false;
	}
	
	// initialize the image structure
	image->Init(data, width, height, channels);
	if (!image->IsValid()) {
		std::cerr << "Invalid image data after loading." << std::endl;
		image->Clear();
		return false;
	}
	std::cout << "Loaded image from " << filename << " with dimensions " << image->width << "x" << image->height << " and " << image->channels << " channels." << std::endl;
	return true;
}

int core::ImageUtils::SaveImage(const std::string& filename, Image *image) {
	if (image->channels < 0 || image->channels > 4) {
		std::cerr << "Unsupported number of channels: " << image->channels << std::endl; 
		return 0;
	}
	if (image->width <= 0 || image->height <= 0) {
		std::cerr << "Invalid image dimensions: " << image->width << "x" << image->height << std::endl;
		return 0;
	}
	if (!image->data) {
		std::cerr << "Image data is null." << std::endl;
		return 0;
	}
	std::cout << "Saving image to " << filename << " with dimensions " <<image-> width << "x" << image->height << " and " << image->channels << " channels." << std::endl;
	return stbi_write_png(filename.c_str(), image->width, image->height, image->channels, image->data, image->width * image->channels);
}