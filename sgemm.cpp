#include "runner.hpp"

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
constexpr const char *kErrLogFile = "matrixValidationFailure.txt";
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    if (argc != 2) {
      std::cerr << "Please select a kernel (range 0 - 12, 0 for reference)"
                << std::endl;
      Kokkos::finalize();
      return EXIT_FAILURE;
    }

    const int kernel_num = std::stoi(argv[1]);
    if (kernel_num < 0 || kernel_num > 12) {
      std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
      Kokkos::finalize();
      return EXIT_FAILURE;
    }

    std::cout << "Running kernel " << kernel_num << " on execution space "
              << sgemm_kokkos::ExecSpace::name() << "." << std::endl;

    const std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
    const int max_size = sizes[sizes.size() - 1];

    const float alpha = 0.5F; // GEMM input parameters
    const float beta = 3.0F;
    constexpr int repeat_times = 50;

    sgemm_kokkos::Matrix dA("A", max_size, max_size);
    sgemm_kokkos::Matrix dB("B", max_size, max_size);
    sgemm_kokkos::Matrix dC("C", max_size, max_size);
    sgemm_kokkos::Matrix dC_ref("C_ref", max_size, max_size);
    sgemm_kokkos::Matrix dC_init("C_init", max_size, max_size);

    auto hA = Kokkos::create_mirror_view(dA);
    auto hB = Kokkos::create_mirror_view(dB);
    auto hC = Kokkos::create_mirror_view(dC_init);

    sgemm_kokkos::randomize_matrix(hA.data(), max_size * max_size);
    sgemm_kokkos::randomize_matrix(hB.data(), max_size * max_size);
    sgemm_kokkos::randomize_matrix(hC.data(), max_size * max_size);

    Kokkos::deep_copy(dA, hA);
    Kokkos::deep_copy(dB, hB);
    Kokkos::deep_copy(dC_init, hC);

    for (const int size : sizes) {
      const int m = size;
      const int n = size;
      const int k = size;

      Kokkos::deep_copy(dC, dC_init);
      Kokkos::deep_copy(dC_ref, dC_init);

      std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
                << ", beta: " << beta << std::endl;

      if (kernel_num != 0) {
        sgemm_kokkos::run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref);
        sgemm_kokkos::run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta,
                                 dC);

        const auto hOut =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dC);
        const auto hRef =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dC_ref);

        if (!sgemm_kokkos::verify_matrix(hRef.data(), hOut.data(), m, n,
                                         max_size)) {
          std::cout << "Failed to pass correctness verification against "
                       "reference kernel."
                    << std::endl;
          if (m <= 128) {
            std::cout << " Logging faulty output into " << kErrLogFile
                      << "\n";
            std::ofstream fs(kErrLogFile);
            fs << "A:\n";
            sgemm_kokkos::print_matrix(hA.data(), m, k, max_size, fs);
            fs << "B:\n";
            sgemm_kokkos::print_matrix(hB.data(), k, n, max_size, fs);
            fs << "C:\n";
            sgemm_kokkos::print_matrix(hOut.data(), m, n, max_size, fs);
            fs << "Should:\n";
            sgemm_kokkos::print_matrix(hRef.data(), m, n, max_size, fs);
          }
          Kokkos::finalize();
          return EXIT_FAILURE;
        }
      }

      Kokkos::Timer timer;
      for (int rep = 0; rep < repeat_times; ++rep) {
        sgemm_kokkos::run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta,
                                 dC);
      }
      Kokkos::fence();
      const double elapsed_time = timer.seconds();

      const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                           static_cast<double>(k);
      std::printf(
          "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. "
          "size: (%d).\n",
          elapsed_time / repeat_times,
          (repeat_times * flops * 1.0e-9) / elapsed_time, m);
      std::fflush(stdout);
    }
  }

  Kokkos::finalize();
  return EXIT_SUCCESS;
}
