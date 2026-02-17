#pragma once

#include "common.hpp"

namespace sgemm_kokkos::kernels {

template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN,
          int NUM_THREADS, typename ViewA, typename ViewB, typename ViewC>
void sgemmDoubleBuffering(const int M, const int N, const int K,
                          const float alpha, const ViewA &A, const ViewB &B,
                          const float beta, const ViewC &C) {
  static_assert(WM > 0 && WN > 0 && WNITER > 0 && NUM_THREADS > 0);
  sgemm_2d_blocktiling<BM, BN, BK, TM, TN>(M, N, K, alpha, A, B, beta, C,
                                           "sgemmDoubleBuffering");
}

} // namespace sgemm_kokkos::kernels
