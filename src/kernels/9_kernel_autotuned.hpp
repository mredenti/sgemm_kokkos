#pragma once

#include "common.hpp"

namespace sgemm_kokkos::kernels {

template <int BM, int BN, int BK, int TM, int TN, typename ViewA,
          typename ViewB, typename ViewC>
void sgemmAutotuned(const int M, const int N, const int K, const float alpha,
                    const ViewA &A, const ViewB &B, const float beta,
                    const ViewC &C) {
  sgemm_2d_blocktiling<BM, BN, BK, TM, TN>(M, N, K, alpha, A, B, beta, C,
                                           "sgemmAutotuned");
}

} // namespace sgemm_kokkos::kernels
