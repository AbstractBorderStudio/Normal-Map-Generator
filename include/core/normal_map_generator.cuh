#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#include <image_utils.h>
#include <linmath.h>
#include <algorithm>

#define BLOCK_SIZE 16

namespace core {
    class NormalMapGenerator {
    private:
        unsigned char *inputData = nullptr;
        unsigned char *outputData = nullptr;
        int inputBytes = 0;
        int outputBytes = 0;
        
        void LoadImageDataToDevice(Image *inputImage);
        void CopyOutputDataToHost(unsigned char* outputData);
        void ClearDeviceMemory();
    public:
        NormalMapGenerator() = default;
        ~NormalMapGenerator() = default;
        void GenerateNormalMapGPU(Image* inputImage, Image* outputImage, float strength, int optimizationType, bool addPadding, bool useCornerPixels);
        void GenerateNormalMapCPU(Image* inputImage, Image* outputImage, float strength = 1.0f);
    };
}