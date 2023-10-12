#include "config.hpp"
#include <vt/transport.h>

namespace Jacobi {
Config::Config(int argc, char** argv){
  for(int i = 0; i < argc; i++){
    std::string arg = argv[i];
    if(       arg == "--decomp"){
      int x = std::stoi(argv[++i]);
      int y = std::stoi(argv[++i]);
      int z = std::stoi(argv[++i]);
      colRange = vt::Index3D(x,y,z);
    } else if(arg == "--input"){
      int x = std::stoi(argv[++i]);
      int y = std::stoi(argv[++i]);
      int z = std::stoi(argv[++i]);
      dataRange = vt::Index3D(x,y,z);
    } else if(arg == "--max-iters") {
      maxIter = std::stoi(argv[++i]);
    } else if(arg == "--tolerance") {
      tolerance = std::stod(argv[++i]);
    } else if(arg == "--async-serialize") {
      asyncCheckpoint = true;
    }
  }

  /* --- Print information about the simulation */
  if(vt::theContext()->getNode() == 0){
    fmt::print(
      stdout, "\n - Solve the linear system for the Laplacian with homogeneous Dirichlet"
      " on [0, 1] x [0, 1] x [0, 1]\n"
    );
    fmt::print(" - Second-order centered finite difference\n");
    fmt::print(" - {} elements decomposed onto {} objects.\n", dataRange.toString(), colRange.toString());
    fmt::print(" - Maximum number of iterations {}\n", maxIter);
    fmt::print(" - Convergence tolerance {}\n", tolerance);
    fmt::print("\n");
  }
}
}

ResilienceConfig::ResilienceConfig(int argc, char** argv, Jacobi::Config app_cfg){
  for(int i = 0; i < argc; i++){
    std::string arg = argv[i];
    if(arg == "--config")
      config_filename = argv[++i];
    else if(arg == "--mode")
      context_mode = argv[++i];
    else if(arg == "--freq")
      checkpoint_frequency = std::stoi(argv[++i]);
    else if(arg == "--kill")
      kill_iter = std::stoi(argv[++i]);
    else if(arg == "--kill-rank")
      kill_rank = std::stoi(argv[++i]);
    else if(arg == "--iters-per-phase")
      iters_per_phase = std::stoi(argv[++i]);
    else if(arg == "--iters-per-epoch")
      iters_per_epoch = std::stoi(argv[++i]);
  }


  if(context_mode == "VT") {
    if(iters_per_epoch == 0) iters_per_epoch = -1;
    context = kr::make_context(vt::theContext(), config_filename);
  } else if(context_mode == "MPI"){
    if(iters_per_epoch == 0){
      iters_per_epoch = checkpoint_frequency;
      //Can't infer both iters_per_epoch and checkpoint_frequency
      assert(checkpoint_frequency != 0);
    }
    context = kr::make_context(MPI_COMM_WORLD, config_filename);
  } else throw std::invalid_argument("Valid --mode values are VT or MPI");
 
  std::string freq_str;
  if(checkpoint_frequency < 0) {
    freq_str = "never";
    checkpoint_filter = [](int iter){ return false; };
  } else if(checkpoint_frequency == 0){
    freq_str = "according to json";
    checkpoint_filter = context->default_filter();
  } else {
    freq_str = fmt::format("every {} iterations", checkpoint_frequency);
    checkpoint_filter = kr::Filter::NthIterationFilter(checkpoint_frequency);
  }


  if(iters_per_phase < 1) iters_per_phase = app_cfg.maxIter+1;
  if(iters_per_epoch < 1) iters_per_epoch = app_cfg.maxIter+1;


  if(vt::theContext()->getNode() == 0) {
    fmt::print("kr:: {} Context configured against {}\n", context_mode, config_filename);
    fmt::print("kr:: Checkpointing {}\n", freq_str);
    if(kill_iter > 0 && kill_rank > 0){
      fmt::print("Generating failure at iteration {} on rank {}\n", kill_iter, kill_rank);
      if(kill_rank >= vt::theContext()->getNumNodes()){
        fmt::print("WARNING: kill_rank {} does not exist!\n", kill_rank);
      }
    }
    
    if(iters_per_epoch == -1){
      fmt::print("kr:: instructing app not to bound iterations\n");
    } else {
      fmt::print("kr:: instructing app to bound every {} iterations\n", iters_per_epoch);
    }

    if(iters_per_phase == -1){
      fmt::print("kr:: instructing app not to use phases\n");
    } else {
      fmt::print("kr:: instructing app to phase every {} iterations\n", iters_per_phase);
    }
  }
}

void ResilienceConfig::try_kill(int current_iteration){
  if(kill_iter == current_iteration && 
     kill_rank == vt::theContext()->getNode()){
    fmt::print(stderr, "Rank {} simulating failure on iteration {}\n", 
                       kill_rank, kill_iter);
    exit(1);
  }
};
