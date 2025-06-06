#include <normal_map_generator.cuh>

/// @brief 
__shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

#pragma region GPU Normal Map Generation

__device__ float3 normalize(float3 v) {
    float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (length > 0.0f) {
        return make_float3(v.x / length, v.y / length, v.z / length);
    } else {
        return make_float3(0.0f, 0.0f, 0.0f); // Return zero vector if length is zero
    }
}

__global__
void GenerateNormalMapKernel(unsigned char* inputData, unsigned char* outputData, int width, int height, int channels, float strength = 1.0f) {
    // calculate global coordinates
	int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (gx >= width || gy >= height || gx < 0 || gy < 0) {
		return; // Out of bounds
	}
	
	// Helper to get clamped luminance
    auto getLuminance = [&](int idx) -> float {
		return 0.299f * inputData[idx] + 0.587f * inputData[idx + 1] + 0.114f * inputData[idx + 2];
    };
	
	// calculate local coordinates with padding
	int lx = threadIdx.x + 1; 
	int ly = threadIdx.y + 1; 
	int gidx = (gy * width + gx) * channels;

	// load the central pixel luminance into shared memory with padding
	if (gx < width && gy < height) {
		float lum = getLuminance(gidx);
		tile[ly][lx] = lum;
	}

	// load the surrounding pixels into shared memory with padding
	// Left
	if (threadIdx.x == 0 && gx > 0) {
		int idx = (gy * width + (gx - 1)) * channels;
		tile[ly][lx - 1] = 0.299f * inputData[idx] + 0.587f * inputData[idx+1] + 0.114f * inputData[idx+2];
	}

	// Right
	if (threadIdx.x == blockDim.x - 1 && gx < width - 1) {
		int idx = (gy * width + (gx + 1)) * channels;
		tile[ly][lx + 1] = 0.299f * inputData[idx] + 0.587f * inputData[idx+1] + 0.114f * inputData[idx+2];
	}

	// Top
	if (threadIdx.y == 0 && gy > 0) {
		int idx = ((gy - 1) * width + gx) * channels;
		tile[ly - 1][lx] = 0.299f * inputData[idx] + 0.587f * inputData[idx+1] + 0.114f * inputData[idx+2];
	}

	// Bottom
	if (threadIdx.y == blockDim.y - 1 && gy < height - 1) {
		int idx = ((gy + 1) * width + gx) * channels;
		tile[ly + 1][lx] = 0.299f * inputData[idx] + 0.587f * inputData[idx+1] + 0.114f * inputData[idx+2];
	}

	// wait for all threads to finish loading data into shared memory
	__syncthreads();

	// use the shaded memory to calculate the normal map
	float left   = tile[ly][lx - 1];
	float right  = tile[ly][lx + 1];
	float top    = tile[ly - 1][lx];
	float bottom = tile[ly + 1][lx];

	float dx = (right - left) * strength;
	float dy = (bottom - top) * strength;

	float3 normal = normalize(make_float3(-dx, -dy, 1.0f));
    outputData[gidx + 0] = static_cast<unsigned char>((normal.x * 0.5f + 0.5f) * 255);
    outputData[gidx + 1] = static_cast<unsigned char>((normal.y * 0.5f + 0.5f) * 255);
    outputData[gidx + 2] = static_cast<unsigned char>((normal.z * 0.5f + 0.5f) * 255);
    // Preserve alpha channel if present
	if (channels == 4)
        outputData[gidx + 3] = inputData[gidx + 3]; 
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
	err = cudaMemcpy(inputData, image->data.data(), inputBytes, cudaMemcpyHostToDevice);
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

/// @brief Generates a normal map from the input image using GPU.
/// @param inputImage Input image to generate the normal map from.
/// @param outputImage Normal map output image.
/// @param strength Strength of the normal map generation.
void core::NormalMapGenerator::GenerateNormalMapGPU(Image* inputImage, Image* outputImage, float strength) {
	if (!inputImage->IsValid()) {
		return;
	}

	// Load input image data to device
	LoadImageDataToDevice(inputImage);
	if (inputData == nullptr || outputData == nullptr) {
		throw std::runtime_error("Device memory not allocated for input or output data.");
	}
	
	// Allocate memory for the output image data
	unsigned char* result = (unsigned char*)malloc(inputImage->width * inputImage->height * inputImage->channels * sizeof(unsigned char));
	
	// Compute grid and block sizes
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridSize((inputImage->width + blockSize.x - 1) / blockSize.x, 
					(inputImage->height + blockSize.y - 1) / blockSize.y);

	// Launch the kernel to generate the normal map
	GenerateNormalMapKernel<<<gridSize, blockSize>>>(inputData, outputData, inputImage->width, inputImage->height, inputImage->channels, strength);
	cudaDeviceSynchronize();
	
	// Copy the output data from device to host
	CopyOutputDataToHost(result);
	if (result == nullptr) {
		throw std::runtime_error("Failed to copy output data to host.");
	}

	outputImage->Init("", result, inputImage->width, inputImage->height, inputImage->channels);
	ClearDeviceMemory();
}
#pragma endregion


#pragma region CPU Normal Map Generation
/// @brief Generates a normal map from the input image using CPU.
/// @param inputImage Input image to generate the normal map from.
/// @param outputImage Normal map output image.
/// @param strength Strength of the normal map generation.
void core::NormalMapGenerator::GenerateNormalMapCPU(Image* inputImage, Image* outputImage, float strength) {
	if (!inputImage || !inputImage->IsValid()) return;
	
    int width = inputImage->width;
    int height = inputImage->height;
    int channels = inputImage->channels;
    const unsigned char* in = inputImage->data.data();

    std::vector<unsigned char> out(width * height * channels);

	auto clamp = [&](int v, int minv, int maxv) -> int {
		return std::max(minv, std::min(v, maxv));
	};

	// Helper to clamp values
    auto getLuminance = [&](int x, int y) -> float {
		x = clamp(x, 0, inputImage->width - 1);
		y = clamp(y, 0, inputImage->height - 1);
		int idx = (y * inputImage->width + x) * inputImage->channels;
		return 0.299f * in[idx] + 0.587f * in[idx + 1] + 0.114f * in[idx + 2];
	};

	// Loop through each pixel to calculate the normal map

	for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float left   = getLuminance(x - 1, y);
            float right  = getLuminance(x + 1, y);
            float top    = getLuminance(x, y - 1);
            float bottom = getLuminance(x, y + 1);

            float dx = (right - left) * strength;
            float dy = (bottom - top) * strength;

            float nx = -dx;
            float ny = -dy;
            float nz = 1.0f;

            float len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len > 0.0f) {
                nx /= len;
                ny /= len;
                nz /= len;
            }

            int idx = (y * width + x) * channels;
            out[idx + 0] = static_cast<unsigned char>((nx * 0.5f + 0.5f) * 255);
            out[idx + 1] = static_cast<unsigned char>((ny * 0.5f + 0.5f) * 255);
            out[idx + 2] = static_cast<unsigned char>((nz * 0.5f + 0.5f) * 255);
            if (channels == 4)
                out[idx + 3] = in[idx + 3]; // preserve alpha
        }
    }

    outputImage->Init("", out.data(), width, height, channels);
}
#pragma endregion