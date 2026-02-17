cmake_minimum_required(VERSION 3.22)
project(NVIDIA_SGEMM_PRACTICE_KOKKOS LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)
check_language(CUDA)

option(SGEMM_KOKKOS_ENABLE_CUDA "Enable CUDA backend in Kokkos when available" ON)

if(SGEMM_KOKKOS_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
else()
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
endif()

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
  Kokkos
  URL https://github.com/kokkos/kokkos/archive/refs/tags/5.0.0.zip
)
FetchContent_MakeAvailable(Kokkos)

add_executable(
  sgemm
  sgemm.cpp
  src/runner.cpp
)

target_include_directories(sgemm PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(sgemm PRIVATE Kokkos::kokkos)
