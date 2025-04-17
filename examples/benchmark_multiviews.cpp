/*
 *
 *                        Kokkos v. 3.0
 *       Copyright (2020) National Technology & Engineering
 *               Solutions of Sandia, LLC (NTESS).
 *
 * Under the terms of Contract DE-NA0003525 with NTESS,
 * the U.S. Government retains certain rights in this software.
 *
 * Kokkos is licensed under 3-clause BSD terms of use:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the Corporation nor the names of the
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Questions? Contact Christian R. Trott (crtrott@sandia.gov)
 */

#include <CLI/CLI.hpp>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <resilience/Resilience.hpp>
#include <random>

#include <iostream>

int main(int argc, char* argv[]) {
  int rank, nbProcs, nbLines, M;
  double wtime, memSize;

  CLI::App app{"A benchmark for testing view registration scaling"};

  std::size_t mem_size    = 100;
  std::size_t nsteps      = 600;
  double precision        = 0.00001;
  int chk_interval        = 100;
  std::string config_path = "config.json";
  std::size_t num_views   = 1;
  std::string scale       = "weak";
  app.add_option("-s,--size", mem_size, "Problem size");
  app.add_option("-n,--nsteps", nsteps, "Number of timesteps");
  app.add_option("-p,--precision", precision, "Min precision");
  app.add_option("-c,--checkpoint-interval", chk_interval,
                 "Checkpoint interval");
  app.add_option("--config", config_path, "Config file");
  app.add_option("--views", num_views, "Number of Kokkos Views");
  app.add_option("--scale", scale, "Weak or strong scaling");

  CLI11_PARSE(app, argc, argv);

  int strong = 0;
  std::cout << "NUM VIEWS " << num_views << '\n';
  if (scale == "strong") {
    strong = 1;
  } else {
    strong = 0;
  }

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nbProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (mem_size == 0) {
    printf("Wrong memory size! See usage\n");
    exit(3);
  }

  Kokkos::initialize(argc, argv);
  {
    auto ctx = KokkosResilience::make_context(MPI_COMM_WORLD, config_path);

    const auto filt =
        KokkosResilience::Filter::NthIterationFilter(chk_interval);

    if (!strong) {
      /* weak scaling */
      M       = (int)sqrt((double)(mem_size * 1024.0 * 1024.0 * nbProcs) /
                          (2 * sizeof(double)));  // two matrices needed
      nbLines = (M / nbProcs) + 3;
    } else {
      /* strong scaling */
      M       = (int)sqrt((double)(mem_size * 1024.0 * 1024.0 * nbProcs) /
                          (2 * sizeof(double) * nbProcs));  // two matrices needed
      nbLines = (M / nbProcs) + 3;
    }

    std::vector<KokkosResilience::View<double*>> MyViews(num_views);
    for (std::size_t i = 0; i < num_views; ++i) {
      Kokkos::resize(MyViews[i], M * nbLines);
      Kokkos::deep_copy(MyViews[i], 1.0);
    }

    memSize = num_views * M * nbLines * sizeof(double) / (1024 * 1024);

    if (rank == 0) {
      if (!strong) {
        std::cout << "Local data size is " << M << " x " << nbLines << " = "
                  << memSize << " MB (" << mem_size << ") " << num_views
                  << " Views.\n";
      } else {
        std::cout << "Local data size is " << M << " x " << nbLines << " = "
                  << memSize << " MB (" << mem_size / nbProcs << ") "
                  << num_views << " Views.\n";
      }
      std::cout << "Target precision: " << precision << '\n';
      std::cout << "Maximum number of iterations: " << nsteps << '\n';
      std::cout << "Array size: " << M * nbLines << '\n';
    }

    wtime         = MPI_Wtime();
    std::size_t i = 1 + KokkosResilience::latest_version(*ctx, "test_kokkos");

    while (i < nsteps) {
      KokkosResilience::checkpoint(
          *ctx, "test_kokkos", i,
          [=]() {  // Nic, tell me what should I put for []/
            for (std::size_t j = 0; j < num_views; ++j) {
              Kokkos::parallel_for(
                  M * nbLines, KOKKOS_LAMBDA(const int& k) {
                    MyViews[j](k) = (double)k * (double)(k + 1);
                  });
            }
          },
          filt);

      i++;
    }
    if (rank == 0)
      printf("Execution finished in %lf seconds.\n", MPI_Wtime() - wtime);
  }
  Kokkos::finalize();

  MPI_Finalize();
  return 0;
}
