#include <app.cuh>

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

int main()
{
    
    // App app = App(SCR_WIDTH, SCR_HEIGHT, "Normal Map Generator");
    
    // // Initialize the application
    // if (!app.Init())
    // {
    //     fprintf(stderr, "Failed to initialize the application.\n");
    //     return -1;
    // }
    // else
    // {
    //     printf("Application initialized successfully.\n");
    // }
    
    // app.Run();

    int width, height, channels;
    unsigned char* data = core::ImageUtils::load_image("../resources/source.png", &width, &height, &channels);
    core::ImageUtils::save_image("../resources/output.png", data, width, height, channels);

    return 0;
}