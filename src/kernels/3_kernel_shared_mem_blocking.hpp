#pragma once

#include "common.hpp"

namespace sgemm_kokkos::kernels {

template <int BLOCKSIZE, typename ViewA, typename ViewB, typename ViewC>
void sgemm_shared_mem_block(const int M, const int N, const int K,
                            const float alpha, const ViewA &A,
                            const ViewB &B, const float beta,
                            const ViewC &C) {
  static_assert(BLOCKSIZE > 0);
  sgemm_tiled<BLOCKSIZE, BLOCKSIZE, BLOCKSIZE>(
      M, N, K, alpha, A, B, beta, C, "sgemm_shared_mem_block");
}

} // namespace sgemm_kokkos::kernels
