#!/bin/bash

module load profile/candidate
module load nvhpc/24.5
module load cuda
module load gcc/12.2.0
module load cmake/4.1.2

# Eventually move this to a CMakePresets.json file
BUILD_DIR="build"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}" && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
cd ..
