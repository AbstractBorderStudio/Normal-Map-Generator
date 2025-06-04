#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <image_utils.h>

bool core::ImageUtils::TryLoadImage(const std::string& filePath, Image *image) {
	// Check if the filePath is valid
	if (filePath.empty()) {
		std::cerr << "Filename is empty." << std::endl;
		return false;
	}

	if (image == nullptr) {
		std::cerr << "Initializing image" << std::endl;
		image = new Image();
	}

	if (!image->IsValid())
	{
		std::cerr << "Image is not valid, clearing and reinitializing." << std::endl;
		image->Clear();
	}

	if (image->imagePath.compare(filePath) == 0) {
		std::cout << "Image already initialized with the same path, skipping reinitialization." << std::endl;
		return false;
	}

	// load the image data
	int width, height, channels;
	unsigned char* data = stbi_load(filePath.c_str(), &width, &height, &channels, 0);
	if (!data) {
		std::cerr << "Failed to load image: " << filePath << " - " << stbi_failure_reason() << std::endl;
		return false;
	}
	
	// Validate image dimensions and channels
	if (width <= 0 || height <= 0) {
		std::cerr << "Invalid image dimensions: " << width << "x" << height << std::endl;
		stbi_image_free(data);
		return false;
	}
	
	// initialize the image structure
	image->Init(filePath, data, width, height, channels);
	if (!image->IsValid()) {
		std::cerr << "Invalid image data." << std::endl;
		image->Clear();
		return false;
	}
	std::cout << "Loaded image from " << filePath << " with dimensions " << image->width << "x" << image->height << " and " << image->channels << " channels." << std::endl;
	free(data);
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
	if (image->data.empty() != 0) {
		std::cerr << "Image data is null." << std::endl;
		return 0;
	}
	std::cout << "Saving image to " << filename << " with dimensions " <<image-> width << "x" << image->height << " and " << image->channels << " channels." << std::endl;
	return stbi_write_png(
		filename.c_str(), 
		image->width, 
		image->height, 
		image->channels, 
		image->data.data(), 
		image->width * image->channels
);
}