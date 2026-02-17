#include "kernels.hpp"
#include "runner.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <stdexcept>
#include <sys/time.h>

namespace sgemm_kokkos {

void randomize_matrix(float *mat, const int size) {
  timeval time{};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < size; ++i) {
    float value = static_cast<float>(rand() % 5) +
                  0.01F * static_cast<float>(rand() % 5);
    value = (rand() % 2 == 0) ? value : -value;
    mat[i] = value;
  }
}

void range_init_matrix(float *mat, const int size) {
  for (int i = 0; i < size; ++i) {
    mat[i] = static_cast<float>(i);
  }
}

void zero_init_matrix(float *mat, const int size) {
  for (int i = 0; i < size; ++i) {
    mat[i] = 0.0F;
  }
}

void copy_matrix(const float *src, float *dst, const int size) {
  for (int i = 0; i < size; ++i) {
    dst[i] = src[i];
  }
}

void print_matrix(const float *matrix, const int rows, const int cols,
                  const int ld, std::ofstream &fs) {
  fs << std::setprecision(2) << std::fixed;
  fs << "[";
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      fs << std::setw(5) << matrix[row * ld + col];
      if (col + 1 < cols) {
        fs << ", ";
      }
    }
    if (row + 1 < rows) {
      fs << ";\n";
    }
  }
  fs << "]\n";
}

bool verify_matrix(const float *reference, const float *output, const int rows,
                   const int cols, const int ld, const float tolerance) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const int idx = row * ld + col;
      const double diff = std::fabs(reference[idx] - output[idx]);
      if (std::isnan(diff) || diff > tolerance) {
        std::printf(
            "Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at (%d, %d)\n",
            reference[idx], output[idx], diff, row, col);
        return false;
      }
    }
  }
  return true;
}

namespace {

void run_reference(const int M, const int N, const int K, const float alpha,
                   const Matrix &A, const Matrix &B, const float beta,
                   const Matrix &C) {
  kernels::sgemm_naive(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_naive(const int M, const int N, const int K, const float alpha,
                     const Matrix &A, const Matrix &B, const float beta,
                     const Matrix &C) {
  kernels::sgemm_naive(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_coalesce(const int M, const int N, const int K,
                        const float alpha, const Matrix &A, const Matrix &B,
                        const float beta, const Matrix &C) {
  kernels::sgemm_global_mem_coalesce<32>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_shared_mem_block(const int M, const int N, const int K,
                                const float alpha, const Matrix &A,
                                const Matrix &B, const float beta,
                                const Matrix &C) {
  kernels::sgemm_shared_mem_block<32>(M, N, K, alpha, A, B, beta, C);
}

void runSgemm1DBlocktiling(const int M, const int N, const int K,
                           const float alpha, const Matrix &A,
                           const Matrix &B, const float beta,
                           const Matrix &C) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 8;
  constexpr int TM = 8;
  kernels::sgemm1DBlocktiling<BM, BN, BK, TM>(M, N, K, alpha, A, B, beta, C);
}

void runSgemm2DBlocktiling(const int M, const int N, const int K,
                           const float alpha, const Matrix &A,
                           const Matrix &B, const float beta,
                           const Matrix &C) {
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;
  if (M >= 128 && N >= 128) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    kernels::sgemm2DBlocktiling<BM, BN, BK, TM, TN>(M, N, K, alpha, A, B, beta,
                                                    C);
  } else {
    constexpr int BM = 64;
    constexpr int BN = 64;
    kernels::sgemm2DBlocktiling<BM, BN, BK, TM, TN>(M, N, K, alpha, A, B, beta,
                                                    C);
  }
}

void runSgemmVectorize(const int M, const int N, const int K,
                       const float alpha, const Matrix &A, const Matrix &B,
                       const float beta, const Matrix &C) {
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;
  if (M >= 128 && N >= 128) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    kernels::sgemmVectorize<BM, BN, BK, TM, TN>(M, N, K, alpha, A, B, beta,
                                                C);
  } else {
    constexpr int BM = 64;
    constexpr int BN = 64;
    kernels::sgemmVectorize<BM, BN, BK, TM, TN>(M, N, K, alpha, A, B, beta,
                                                C);
  }
}

void runSgemmResolveBankConflicts(const int M, const int N, const int K,
                                  const float alpha, const Matrix &A,
                                  const Matrix &B, const float beta,
                                  const Matrix &C) {
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;
  if (M >= 128 && N >= 128) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    kernels::sgemmResolveBankConflicts<BM, BN, BK, TM, TN>(
        M, N, K, alpha, A, B, beta, C);
  } else {
    constexpr int BM = 64;
    constexpr int BN = 64;
    kernels::sgemmResolveBankConflicts<BM, BN, BK, TM, TN>(
        M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmResolveBankExtraCol(const int M, const int N, const int K,
                                 const float alpha, const Matrix &A,
                                 const Matrix &B, const float beta,
                                 const Matrix &C) {
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;
  if (M >= 128 && N >= 128) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    kernels::sgemmResolveBankExtraCol<BM, BN, BK, TM, TN>(M, N, K, alpha, A, B,
                                                          beta, C);
  } else {
    constexpr int BM = 64;
    constexpr int BN = 64;
    kernels::sgemmResolveBankExtraCol<BM, BN, BK, TM, TN>(M, N, K, alpha, A, B,
                                                          beta, C);
  }
}

void runSgemmAutotuned(const int M, const int N, const int K,
                       const float alpha, const Matrix &A, const Matrix &B,
                       const float beta, const Matrix &C) {
  constexpr int BK = 16;
  constexpr int TM = 8;
  constexpr int TN = 8;
  constexpr int BM = 128;
  constexpr int BN = 128;
  kernels::sgemmAutotuned<BM, BN, BK, TM, TN>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmWarptiling(const int M, const int N, const int K,
                        const float alpha, const Matrix &A, const Matrix &B,
                        const float beta, const Matrix &C) {
  constexpr int NUM_THREADS = 128;
  constexpr int BN = 128;
  constexpr int BM = 128;
  constexpr int BK = 16;
  constexpr int WN = 64;
  constexpr int WM = 64;
  constexpr int WNITER = 4;
  constexpr int TN = 4;
  constexpr int TM = 8;
  kernels::sgemmWarptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>(
      M, N, K, alpha, A, B, beta, C);
}

void runSgemmDoubleBuffering(const int M, const int N, const int K,
                             const float alpha, const Matrix &A,
                             const Matrix &B, const float beta,
                             const Matrix &C) {
  constexpr int NUM_THREADS = 256;
  constexpr int BN = 256;
  constexpr int BM = 128;
  constexpr int BK = 16;
  constexpr int WN = 32;
  constexpr int WM = 128;
  constexpr int WNITER = 1;
  constexpr int TN = 8;
  constexpr int TM = 8;
  kernels::sgemmDoubleBuffering<BM, BN, BK, WM, WN, WNITER, TM, TN,
                                NUM_THREADS>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmDoubleBuffering2(const int M, const int N, const int K,
                              const float alpha, const Matrix &A,
                              const Matrix &B, const float beta,
                              const Matrix &C) {
  constexpr int NUM_THREADS = 128;
  constexpr int BN = 128;
  constexpr int BM = 128;
  constexpr int BK = 16;
  constexpr int WN = 64;
  constexpr int WM = 64;
  constexpr int WNITER = 4;
  constexpr int TN = 4;
  constexpr int TM = 8;
  kernels::runSgemmDoubleBuffering2<BM, BN, BK, WM, WN, WNITER, TM, TN,
                                    NUM_THREADS>(M, N, K, alpha, A, B, beta,
                                                 C);
}

} // namespace

void run_kernel(const int kernel_num, const int m, const int n, const int k,
                const float alpha, const Matrix &A, const Matrix &B,
                const float beta, const Matrix &C) {
  switch (kernel_num) {
  case 0:
    run_reference(m, n, k, alpha, A, B, beta, C);
    break;
  case 1:
    run_sgemm_naive(m, n, k, alpha, A, B, beta, C);
    break;
  case 2:
    run_sgemm_coalesce(m, n, k, alpha, A, B, beta, C);
    break;
  case 3:
    run_sgemm_shared_mem_block(m, n, k, alpha, A, B, beta, C);
    break;
  case 4:
    runSgemm1DBlocktiling(m, n, k, alpha, A, B, beta, C);
    break;
  case 5:
    runSgemm2DBlocktiling(m, n, k, alpha, A, B, beta, C);
    break;
  case 6:
    runSgemmVectorize(m, n, k, alpha, A, B, beta, C);
    break;
  case 7:
    runSgemmResolveBankConflicts(m, n, k, alpha, A, B, beta, C);
    break;
  case 8:
    runSgemmResolveBankExtraCol(m, n, k, alpha, A, B, beta, C);
    break;
  case 9:
    runSgemmAutotuned(m, n, k, alpha, A, B, beta, C);
    break;
  case 10:
    runSgemmWarptiling(m, n, k, alpha, A, B, beta, C);
    break;
  case 11:
    runSgemmDoubleBuffering(m, n, k, alpha, A, B, beta, C);
    break;
  case 12:
    runSgemmDoubleBuffering2(m, n, k, alpha, A, B, beta, C);
    break;
  default:
    throw std::invalid_argument("Unknown kernel number");
  }

  Kokkos::fence();
}

} // namespace sgemm_kokkos
