cmake_minimum_required(VERSION 3.18)

# project settings
project(NormalMapGenerator VERSION 0.0.1 LANGUAGES CXX CUDA)

# set c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# setup glfw build options
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

# add subdirectories for third-party libraries
add_subdirectory(include/thirdparty/glfw-3.4)  # window manager
add_subdirectory(include/thirdparty/glad)	     # opengl loader
add_subdirectory(include/thirdparty/imgui)     # user interface        

# include header-only libraries
include_directories(${CMAKE_SOURCE_DIR}/include/core)
include_directories(${CMAKE_SOURCE_DIR}/include/thirdparty/stb)
include_directories(${CMAKE_SOURCE_DIR}/include/thirdparty/linmath/include)

# include source files
file(GLOB_RECURSE SRC_FILES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.cu"
)

if (EXISTS "${CMAKE_SOURCE_DIR}/settings")
  file(
    COPY "${CMAKE_SOURCE_DIR}/settings/imgui.ini"
    DESTINATION "${CMAKE_BINARY_DIR}/Debug"
  )
endif()

# create executable
add_executable(${PROJECT_NAME} ${SRC_FILES})

# link libraries
target_link_libraries(${CMAKE_PROJECT_NAME}
    PRIVATE
        imgui     # include glfw e glad (PUBLIC is needed in imgui cmakelists)
)