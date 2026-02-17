#pragma once

#include <Kokkos_Core.hpp>

#include <fstream>

namespace sgemm_kokkos {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using Matrix = Kokkos::View<float **, Kokkos::LayoutRight, ExecSpace>;

void randomize_matrix(float *mat, int size);
void range_init_matrix(float *mat, int size);
void zero_init_matrix(float *mat, int size);
void copy_matrix(const float *src, float *dst, int size);

void print_matrix(const float *matrix, int rows, int cols, int ld,
                  std::ofstream &fs);
bool verify_matrix(const float *reference, const float *output, int rows,
                   int cols, int ld, float tolerance = 0.01F);

void run_kernel(int kernel_num, int m, int n, int k, float alpha,
                const Matrix &A, const Matrix &B, float beta, const Matrix &C);

} // namespace sgemm_kokkos
