#pragma once

#include "common.hpp"

namespace sgemm_kokkos::kernels {

template <int BM, int BN, int BK, int TM, typename ViewA, typename ViewB,
          typename ViewC>
void sgemm1DBlocktiling(const int M, const int N, const int K,
                        const float alpha, const ViewA &A, const ViewB &B,
                        const float beta, const ViewC &C) {
  sgemm_1d_blocktiling<BM, BN, BK, TM>(M, N, K, alpha, A, B, beta, C,
                                       "sgemm1DBlocktiling");
}

} // namespace sgemm_kokkos::kernels
