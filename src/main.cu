#include <app.cuh>

// settings
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

int main()
{
    
    App app = App(SCR_WIDTH, SCR_HEIGHT, "Normal Map Generator");
    
    // Initialize the application
    if (!app.Init())
    {
        fprintf(stderr, "Failed to initialize the application.\n");
        return -1;
    }
    app.Run();
    return 0;
}

