#include <normal_map_generator.cuh>

__device__
void GenerateHeightMap(unsigned char* inputData, unsigned char* outputData, int width, int height, int pixelIndex)
{
	// compute luminnace as heightmap
	float luminance = 0.299f * inputData[pixelIndex] + 
					  0.587f * inputData[pixelIndex + 1] + 
					  0.114f * inputData[pixelIndex + 2];

	// output pixel RGB values)
	outputData[pixelIndex] 		= static_cast<unsigned char>(luminance); // R
	outputData[pixelIndex + 1] 		= static_cast<unsigned char>(luminance); // R
	outputData[pixelIndex + 2] 		= static_cast<unsigned char>(luminance); // R
	outputData[pixelIndex + 3] 		= 255; // R
}

__device__
void SobelGradients(unsigned char* heightMap, unsigned char* outputData, int width, int height, int channels, int pixelIndex) {
	
}

__global__
void GenerateNormalMapKernel(unsigned char* inputData, unsigned char* outputData, int width, int height, int channels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int pixelIndex = (y * width + x) * channels; // Assuming RGB format for output

	if (x < width && y < height) {
		GenerateHeightMap(inputData, outputData, width, height, pixelIndex);
		//SobelGradients(outputData, outputData, width, height, channels, pixelIndex);
	}
}

void core::NormalMapGenerator::LoadImageDataToDevice(Image* image)
{
	if (image == nullptr || !image->IsValid()) {
		throw std::runtime_error("Invalid image data provided to NormalMapGenerator.");
	}

	// Clear previous device memory
	ClearDeviceMemory();

	// Allocate memory on the device for input and output data
	inputBytes = image->width * image->height * image->channels * sizeof(unsigned char);
	outputBytes = image->width * image->height * image->channels * sizeof(unsigned char);

	cudaError_t err = cudaMalloc((void**)&inputData, inputBytes);
	if (err != cudaSuccess) {
		throw std::runtime_error("Failed to allocate device memory for input data.");
	}

	err = cudaMalloc((void**)&outputData, outputBytes);
	if (err != cudaSuccess) {
		cudaFree(inputData);
		throw std::runtime_error("Failed to allocate device memory for output data.");
	}

	// Copy input image data to device
	err = cudaMemcpy(inputData, image->data, inputBytes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		ClearDeviceMemory();
		throw std::runtime_error("Failed to copy image data to device.");
	}
}

void core::NormalMapGenerator::CopyOutputDataToHost(unsigned char* result)
{
	if (result == nullptr) {
		throw std::runtime_error("Output data pointer is null.");
	}

	// Copy the output data from device to host
	cudaError_t err = cudaMemcpy(result, this->outputData, outputBytes, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		ClearDeviceMemory();
		throw std::runtime_error("Failed to copy output data from device to host.");
	}
}

void core::NormalMapGenerator::ClearDeviceMemory()
{
	if (inputData) {
		cudaFree(inputData);
		inputData = nullptr;
	}
	if (outputData) {
		cudaFree(outputData);
		outputData = nullptr;
	}
	inputBytes = 0;
	outputBytes = 0;
}

void core::NormalMapGenerator::GenerateNormalMap(Image* inputImage, Image* outputImage) {
	if (!inputImage->IsValid()) {
		return;
	}

	// Launch the kernel to generate the normal map
	dim3 blockSize(16, 16);
	dim3 gridSize((inputImage->width + blockSize.x - 1) / blockSize.x, (inputImage->height + blockSize.y - 1) / blockSize.y);

	LoadImageDataToDevice(inputImage);
	if (inputData == nullptr || outputData == nullptr) {
		throw std::runtime_error("Device memory not allocated for input or output data.");
	}

	unsigned char* result = (unsigned char*)malloc(inputImage->width * inputImage->height * inputImage->channels * sizeof(unsigned char));
	GenerateNormalMapKernel<<<gridSize, blockSize>>>(inputData, outputData, inputImage->width, inputImage->height, inputImage->channels);
	cudaDeviceSynchronize();
	
	CopyOutputDataToHost(result);
	if (result == nullptr) {
		throw std::runtime_error("Failed to copy output data to host.");
	}

	outputImage->Init("", result, inputImage->width, inputImage->height, inputImage->channels);
}