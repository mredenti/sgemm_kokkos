#pragma once

#include <Kokkos_Core.hpp>

namespace sgemm_kokkos::kernels {

template <int BLOCKSIZE, typename ViewA, typename ViewB, typename ViewC>
void sgemm_global_mem_coalesce(const int M, const int N, const int K,
                               const float alpha, const ViewA &A,
                               const ViewB &B, const float beta,
                               const ViewC &C) {
  static_assert(BLOCKSIZE > 0);
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using Policy = Kokkos::RangePolicy<ExecSpace>;

  Kokkos::parallel_for(
      "sgemm_global_mem_coalesce", Policy(0, M * N),
      KOKKOS_LAMBDA(const int idx) {
        const int row = idx / N;
        const int col = idx % N;

        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
          acc += A(row, k) * B(k, col);
        }
        C(row, col) = alpha * acc + beta * C(row, col);
      });
}

} // namespace sgemm_kokkos::kernels
