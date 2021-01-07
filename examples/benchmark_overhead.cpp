//@HEADER
// ************************************************************************
//     Resilient Extension of Kokkos
//     by Sandia National Laboratories
//
// Sandia National Laboratories is a multimission laboratory managed
// and operated by National Technology and Engineering Solutions of Sandia,
// LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
// U.S. Department of Energy's National Nuclear Security Administration under
// contract DE-NA0003525.
//
// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
// ************************************************************************
//@HEADER


#include <cxxopts/cxxopts.hpp>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <resilience/Resilience.hpp>
#include <random>
#include <resilience/CheckpointFilter.hpp>

using namespace heatdis;

/*
    This sample application is based on the heat distribution code
    originally developed within the FTI project: github.com/leobago/fti
*/

int main(int argc, char *argv[]) {
   int rank, nbProcs, nbLines, M;
   double wtime, memSize, localerror, globalerror = 1;

   auto options = cxxopts::Options("heatdis", "Sample heat distribution code");
   options.add_options()
           ("s,size", "Problem size", cxxopts::value<std::size_t>()->default_value("100"))
           ("n,nsteps", "Number of timesteps", cxxopts::value<std::size_t>()->default_value("600"))
           ("p,precision", "Min precision", cxxopts::value<double>()->default_value("0.00001"))
           ("c,checkpoint-interval", "Checkpoint interval", cxxopts::value<int>()->default_value("100"))
           ("config", "Config file", cxxopts::value<std::string>())
           ("scale", "Weak or strong scaling", cxxopts::value<std::string>())
           ("views", "Number of Kokkos Views", value<std::size_t>()->default_value("1"))
           ;

   options.parse_positional({"config"});
   auto args = options.parse(argc, argv);

   std::size_t nsteps = args["nsteps"].as< std::size_t >();
   const auto precision = args["precision"].as< double >();
   const auto chk_interval = args["checkpoint-interval"].as< int >();
   const auto num_views =  args["views"].as< std::size_t >();
   int strong, str_ret;

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

      std::vector<Kokkos::View<double*> MyViews()
      for( int i = 0 ; i < num_views; ++i )
      {
         Kokkos::resize( MyViews[i], M*nbLines);
         Kokkos::deep_copy(MyViews[i],1.0);
      }



      memSize = num_views * M * nbLines * sizeof(double) / (1024 * 1024);

      if (rank == 0) {
         if (!strong) {
            printf("Local data size is %d x %d = %f MB (%lu) %d Views.\n", M, nbLines, memSize, mem_size, num_views);
         } else {
            printf("Local data size is %d x %d = %f MB (%lu) %d Views.\n", M, nbLines, memSize, mem_size / nbProcs,
                   num_views);
         }
         printf("Target precision : %f \n", precision);
         printf("Maximum number of iterations : %lu \n", nsteps);
      }

      wtime = MPI_Wtime();
      int i = 1 + KokkosResilience::latest_version(*ctx, "test_kokkos");

      while(i < nsteps) {

         KokkosResilience::checkpoint(*ctx, "test_kokkos", i, [&localerror, i, &globalerror,
                 g_view, h_view, nbProcs, rank, M, nbLines]() {

            for( int j = 0; j < num_views; ++j )
            {

               Kokkos::parallel_for( M*nbLines, KOKKOS_LAMBDA (const int& k )
               {
                  MyView[j](k) = (double) rand()/RAND_MAX;
               }
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
