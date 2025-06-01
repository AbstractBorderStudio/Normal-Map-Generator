#include <normal_map_generator.cuh>

__global__ void generateNormalMapKernel(float* image, float* normalMap, int width, int height, float scale) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int idx = y * width + x;

}

void normgenerateNormalMap(float* image, float* normalMap, int width, int height, float scale) {
	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	generateNormalMapKernel<<<gridSize, blockSize>>>(image, normalMap, width, height, scale);
	cudaDeviceSynchronize();
}