cmake_minimum_required(VERSION 3.21)
project(fluid_dynamics LANGUAGES CXX CUDA)

include(FetchContent)

FetchContent_Declare(
	glm
	GIT_REPOSITORY	https://github.com/g-truc/glm.git
	GIT_TAG 	bf71a834948186f4097caa076cd2663c69a10e1e #refs/tags/1.0.1
)

FetchContent_MakeAvailable(glm)

find_package(OpenMP)

file(GLOB_RECURSE SOURCES src/*.cu)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE raylib glm::glm)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()