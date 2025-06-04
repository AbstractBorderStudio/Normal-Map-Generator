#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <image_utils.h>

void core::Image::Init(std::string imagePath, unsigned char* data, int width, int height, int channels)
{
	this->imagePath = imagePath;
	this->data.assign(data, data + (width * height * channels));
	this->width = width;
	this->height = height;
	this->channels = channels;
	this->textureID = -1;
	
	
	GLint maxTexSize = 0;
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTexSize);

	int newWidth = width;
	int newHeight = height;
	std::vector<unsigned char> resizedData;

	if (width > maxTexSize || height > maxTexSize) {
		float scale = std::min((float)maxTexSize / width, (float)maxTexSize / height);
		newWidth = static_cast<int>(width * scale);
		newHeight = static_cast<int>(height * scale);
		resizedData.resize(newWidth * newHeight * channels);

		stbir_resize_uint8(
			data, width, height, 0,
			resizedData.data(), newWidth, newHeight, 0,
			channels
		);
		this->data = std::move(resizedData);
		this->width = newWidth;
		this->height = newHeight;
	} else {
		this->data.assign(data, data + (width * height * channels));
		this->width = width;
		this->height = height;
	}
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
	GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
	glTexImage2D(GL_TEXTURE_2D, 0, format, this->width, this->height, 0, format, GL_UNSIGNED_BYTE, this->data.data());
}

bool core::Image::IsValid()
{
	if (data.empty() || width <= 0 || height <= 0 || channels < 0 || channels > 4) {
		return false;
	}
	return true;
}

void core::Image::UpdateTexture()
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

void core::Image::Clear()
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