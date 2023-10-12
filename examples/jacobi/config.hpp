#ifndef JACOBI_CONFIG_HPP
#define JACOBI_CONFIG_HPP

#include <vt/transport.h>
#include <resilience/Resilience.hpp>

namespace Jacobi {
//Manage solver parameters
struct Config {
  //Number of solver objects to decompose the work into.
  //  --decomp <x> <y> <z>
  vt::Index3D colRange = vt::Index3D(4,4,4);

  //Input size per solver object
  //  --input <x> <y> <z>
  vt::Index3D dataRange = vt::Index3D(50,50,50);

  //Solver stops running after either maxIter iterations, or
  //once tolerance reached.
  //  --tolerance <float>
  //  --max-iters <integer>
  double tolerance = 1e-2;
  int maxIter = 100;

  //Whether solver ought to manage asynchronous checkpointing manually.
  //  --async-serialize
  bool asyncCheckpoint = false;

  Config() = default;

  template<typename SerT>
  void serialize(SerT& s){
    s | colRange | dataRange | tolerance | maxIter | asyncCheckpoint;
  }

  Config(int argc, char** argv);
};
}

//Manage resilience parameters
struct ResilienceConfig {
  //Path to JSON config file for KokkosResilience
  //  --config <path/to/config.json>
  std::string config_filename = "config_jacobi.json";

  //Which type of context to use for consistency-enforcement
  //  --mode <VT or MPI>
  std::string context_mode = "VT";

  //How often to checkpoint, in iterations.
  //  --freq <integer>
  // 0 = from config file
  //-1 = never
  int checkpoint_frequency = 0;
  
  //Where and when to insert a failure. 
  //  --kill <integer>
  int kill_iter = -1;
  //  --kill-rank <integer>
  int kill_rank = 0;

  //How often we should start the next VT phase (requiring an epoch boundary)
  //  --iters-per-phase <integer>
  int iters_per_phase = 30;

  //How often we should arbitrarily insert an epoch boundary to test w/ 
  //some forced synchrony or to ensure correctness with "MPI" context_mode
  //  --iters-per-epoch <integer>
  // 0 = matching checkpoint_frequency if "MPI" context_mode, else never
  //-1 = never
  int iters_per_epoch = 0;

  //Tells context when to checkpoint.
  std::function<bool(int)> checkpoint_filter;
  
  ResilienceConfig(int argc, char** argv, Jacobi::Config app_cfg);

  //Test recovery by exiting if on correct iteration and rank
  void try_kill(int current_iteration);

  //Enable treating this object just like you would the context unique_ptr
  kr::ContextBase* operator->(){ return context.get(); }
  void reset() { context.reset(); }

private:
  std::unique_ptr<kr::ContextBase> context;
};



#endif
