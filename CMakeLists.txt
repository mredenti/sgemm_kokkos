cmake_minimum_required(VERSION 3.24) # Bumped to 3.24 to automatically fix CMP0135 warning
project(NVIDIA_SGEMM_PRACTICE_KOKKOS LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SGEMM_KOKKOS_ENABLE_CUDA "Enable CUDA backend in Kokkos when available" ON)

if(SGEMM_KOKKOS_ENABLE_CUDA)
  # 1. Find the toolkit FIRST. This populates correct paths for the linker.
  find_package(CUDAToolkit REQUIRED)
  
  # 2. Force Shared runtime to avoid static linking errors on NVHPC
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  
  # 3. Now enable the language safely
  enable_language(CUDA)
  set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
  
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ARCH_AMPERE80 ON CACHE BOOL "" FORCE)
else()
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
endif()

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
  Kokkos
  URL https://github.com/kokkos/kokkos/archive/refs/tags/5.0.1.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true # Cleanest way to fix the extraction warning
)
FetchContent_MakeAvailable(Kokkos)

add_executable(
  sgemm
  sgemm.cpp
  src/runner.cpp
)

target_include_directories(sgemm PRIVATE ${PROJECT_SOURCE_DIR}/src)

if(SGEMM_KOKKOS_ENABLE_CUDA)
  # Link Kokkos and explicitly link the CUDA toolkit runtime just to be safe
  target_link_libraries(sgemm PRIVATE Kokkos::kokkos CUDA::cudart)
else()
  target_link_libraries(sgemm PRIVATE Kokkos::kokkos)
endif()