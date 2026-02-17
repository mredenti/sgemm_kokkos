#pragma once

#include <Kokkos_Core.hpp>

namespace sgemm_kokkos::kernels {

template <typename ViewA, typename ViewB, typename ViewC>
void sgemm_naive(const int M, const int N, const int K, const float alpha,
                 const ViewA &A, const ViewB &B, const float beta,
                 const ViewC &C) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using Policy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>;

  Kokkos::parallel_for(
      "sgemm_naive", Policy({0, 0}, {M, N}),
      KOKKOS_LAMBDA(const int row, const int col) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
          acc += A(row, k) * B(k, col);
        }
        C(row, col) = alpha * acc + beta * C(row, col);
      });
}

} // namespace sgemm_kokkos::kernels
