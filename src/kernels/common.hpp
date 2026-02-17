#pragma once

#include <Kokkos_Core.hpp>
#include <string>

namespace sgemm_kokkos::kernels {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using Matrix = Kokkos::View<float **, Kokkos::LayoutRight, ExecSpace>;
using MatrixConst = Kokkos::View<const float **, Kokkos::LayoutRight, ExecSpace>;

KOKKOS_INLINE_FUNCTION
constexpr int ceil_div(const int numerator, const int denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <int BM, int BN, int BK, typename ViewA, typename ViewB, typename ViewC>
void sgemm_tiled(const int M, const int N, const int K, const float alpha,
                 const ViewA &A, const ViewB &B, const float beta,
                 const ViewC &C, const std::string &label) {
  using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
  const int league_m = ceil_div(M, BM);
  const int league_n = ceil_div(N, BN);

  Kokkos::parallel_for(
      label,
      TeamPolicy(league_m * league_n, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const typename TeamPolicy::member_type &team) {
        const int tile_idx = team.league_rank();
        const int tile_m = tile_idx / league_n;
        const int tile_n = tile_idx % league_n;

        const int row0 = tile_m * BM;
        const int col0 = tile_n * BN;

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, BM * BN),
                             [&](const int idx) {
                               const int local_row = idx / BN;
                               const int local_col = idx % BN;
                               const int row = row0 + local_row;
                               const int col = col0 + local_col;

                               if (row >= M || col >= N) {
                                 return;
                               }

                               float acc = 0.0f;
                               for (int k0 = 0; k0 < K; k0 += BK) {
                                 const int kend = (k0 + BK < K) ? (k0 + BK) : K;
                                 for (int k = k0; k < kend; ++k) {
                                   acc += A(row, k) * B(k, col);
                                 }
                               }
                               C(row, col) = alpha * acc + beta * C(row, col);
                             });
      });
}

template <int BM, int BN, int BK, int TM, typename ViewA, typename ViewB,
          typename ViewC>
void sgemm_1d_blocktiling(const int M, const int N, const int K,
                          const float alpha, const ViewA &A, const ViewB &B,
                          const float beta, const ViewC &C,
                          const std::string &label) {
  static_assert(TM > 0);
  sgemm_tiled<BM, BN, BK>(M, N, K, alpha, A, B, beta, C, label);
}

template <int BM, int BN, int BK, int TM, int TN, typename ViewA,
          typename ViewB, typename ViewC>
void sgemm_2d_blocktiling(const int M, const int N, const int K,
                          const float alpha, const ViewA &A, const ViewB &B,
                          const float beta, const ViewC &C,
                          const std::string &label) {
  static_assert(TM > 0 && TN > 0);
  sgemm_tiled<BM, BN, BK>(M, N, K, alpha, A, B, beta, C, label);
}

} // namespace sgemm_kokkos::kernels
