cmake_minimum_required(VERSION 3.18)

project(cfxmine CXX CUDA)

# CUDA 12.8 optimization flags for RTX 5090
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math --ptxas-options=-v")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -DUSE_ROT_ASM_OPT=1")

# Enable newer PTX for Blackwell optimizations (only for nvcc)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90,code=sm_90")
endif()

set(THIRDPARTY_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third-party)

if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
  # Support for wider range of GPU architectures including RTX 5090 Blackwell
  # 61: GTX 1080/1070 (Pascal), 75: RTX 2080/2070 (Turing), 
  # 80: RTX 3080/3090 (Ampere), 86: RTX 3050/3060 (Ampere), 90: RTX 5090 (Blackwell)
  set(CMAKE_CUDA_ARCHITECTURES 61 75 80 86 90)
endif()

find_package(CUDAToolkit)

set(Boost_USE_STATIC_LIBS ON)
# set(Boost_DEBUG ON)
find_package(Boost 1.65.1 REQUIRED COMPONENTS chrono regex system thread)

add_subdirectory(${THIRDPARTY_SOURCE_DIR}/jsoncpp)

add_executable(cfxmine
  src/main.cc
  src/StratumClient.cc
  src/light.cc
  src/sha3.cc
  src/OctopusCPUMiner.cc
  src/OctopusCUDAMiner.cu
)
set_property(TARGET cfxmine PROPERTY CXX_STANDARD 17)
target_include_directories(
  cfxmine
  PUBLIC
  ${THIRDPARTY_SOURCE_DIR}/cxxopts/include
)
target_link_libraries(
  cfxmine
  PUBLIC
  Boost::chrono
  Boost::thread
  Boost::regex
  Boost::system
  CUDA::cudart_static
  jsoncpp_lib
)
