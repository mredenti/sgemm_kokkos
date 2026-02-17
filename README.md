# SGEMM Kokkos Rewrite

This folder is a Kokkos rewrite of the original CUDA SGEMM project.

Goals:
- Keep the same organization and kernel progression (0-12).
- Keep runner flow and benchmark behavior close to the original code.
- Use modern C++20 and Kokkos execution patterns.

## Layout

- `sgemm.cpp`: benchmark entry point and validation loop
- `src/runner.hpp`, `src/runner.cpp`: utilities and kernel dispatch
- `src/kernels.hpp`: includes all numbered kernels
- `src/kernels/*.hpp`: numbered kernels mirroring original progression

## Build

Requirements:
- CMake >= 3.22
- A C++20 compiler

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

Or:

```bash
make build
```

## Run

```bash
./build/sgemm <kernel_num>
```

Kernel range is `0..12`.
