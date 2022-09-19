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


#include <cxxopts.hpp>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <resilience/Resilience.hpp>
#include <random>
#include <resilience/CheckpointFilter.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
   int rank, nbProcs, nbLines, M;
   double wtime, memSize;

   auto options = cxxopts::Options("heatdis", "Sample heat distribution code");
   options.add_options()
           ("s,size", "Problem size", cxxopts::value<std::size_t>()->default_value("100"))
           ("n,nsteps", "Number of timesteps", cxxopts::value<std::size_t>()->default_value("600"))
           ("p,precision", "Min precision", cxxopts::value<double>()->default_value("0.00001"))
           ("c,checkpoint-interval", "Checkpoint interval", cxxopts::value<int>()->default_value("100"))
           ("config", "Config file", cxxopts::value<std::string>())
           ("views", "Number of Kokkos Views", cxxopts::value<std::size_t>()->default_value("1"))
           ("scale", "Weak or strong scaling", cxxopts::value<std::string>()->default_value("weak"))
           ;

   options.parse_positional({"config"});
   auto args = options.parse(argc, argv);

   std::size_t nsteps = args["nsteps"].as< std::size_t >();
   const auto precision = args["precision"].as< double >();
   const auto chk_interval = args["checkpoint-interval"].as< int >();
   const auto num_views =  args["views"].as< std::size_t >();
   int strong              = 0;
   std::cout << "NUM VIEWS " << num_views << '\n';
   std::string scale;
   scale = args["scale"].as< std::string >();
   if (scale == "strong") {
      strong = 1;
   } else {
      strong = 0;
   }


   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nbProcs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   std::size_t mem_size = args["size"].as< std::size_t >();

   if (mem_size == 0) {
      printf("Wrong memory size! See usage\n");
      exit(3);
   }

   Kokkos::initialize(argc, argv);
   {
      auto ctx = KokkosResilience::make_context( MPI_COMM_WORLD, args["config"].as< std::string >() );

      const auto filt = KokkosResilience::Filter::NthIterationFilter( chk_interval );

      if (!strong) {

         /* weak scaling */
         M = (int)sqrt((double)(mem_size * 1024.0 * 1024.0 * nbProcs) / (2 * sizeof(double))); // two matrices needed
         nbLines = (M / nbProcs) + 3;
      } else {

         /* strong scaling */
         M = (int)sqrt((double)(mem_size * 1024.0 * 1024.0 * nbProcs) / (2 * sizeof(double) * nbProcs)); // two matrices needed
         nbLines = (M / nbProcs) + 3;
      }

      std::vector<Kokkos::View<double*>> MyViews(num_views);
      for( int i = 0 ; i < num_views; ++i )
      {
         Kokkos::resize( MyViews[i], M*nbLines);
         Kokkos::deep_copy(MyViews[i],1.0);
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

      wtime = MPI_Wtime();
      int i = 1 + KokkosResilience::latest_version(*ctx, "test_kokkos");

      while(i < nsteps) {

         KokkosResilience::checkpoint(*ctx, "test_kokkos", i, [=]() {   // Nic, tell me what should I put for []/

         for( int j = 0; j < num_views; ++j )
         {

            Kokkos::parallel_for( M*nbLines, KOKKOS_LAMBDA (const int& k )
            {
               MyViews[j](k) = (double)k*(double)(k+1);
            });
         }
       }, filt );

         i++;
      }
      if (rank == 0)
         printf("Execution finished in %lf seconds.\n", MPI_Wtime() - wtime);

   }
   Kokkos::finalize();

   MPI_Finalize();
   return 0;
}
