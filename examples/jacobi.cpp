/*
//@HEADER
// *****************************************************************************
//
//                                jacobi3d_vt.cc
//                       DARMA/vt => Virtual Transport
//
// Copyright 2019-2021 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact darma@sandia.gov
//
// *****************************************************************************
//@HEADER
*/

#include <vt/transport.h>
#include <vt/timing/timing.h>
#include <resilience/Resilience.hpp>
#include <exception>

#include "jacobi.hpp"

const std::string MAIN_LOOP = "Jacobi main loop";

//Manage options for how to checkpoint, when to inject failures
struct ResilienceConfig {
  //See below main for configuration flags
  ResilienceConfig(int argc, char** argv);

  //Test recovery by exiting if on correct iteration and rank
  void try_kill(int current_iteration){
    if(kill_iter == current_iteration && 
       kill_rank == vt::theContext()->getNode()){
      fmt::print(stderr, "Rank {} simulating failure on iteration {}\n", 
                         kill_rank, kill_iter);
      exit(1);
    }
  };


  //Enable treating this object just like you would the context unique_ptr
  kr::ContextBase* operator->(){ return context.get(); }
  void reset() { context.reset(); }

  //Which type of context to use for consistency-enforcement
  std::string context_mode = "VT";

  //How often we should start the next VT phase (requiring an epoch boundary)
  int iters_per_phase = 30;

  //How often we should set an epoch boundary
  //0 = set based on mode; -1 = never
  int iters_per_epoch = 0;

  //Tells context when to checkpoint.
  std::function<bool(int)> checkpoint_filter;

private:
  int kill_iter = -1, kill_rank = 0;
  std::unique_ptr<kr::ContextBase> context;
};

//See jacobi_impl.hpp for command-line configuration options of jacobi.
using JacobiConfig = Jacobi::Config;

int main(int argc, char** argv) {
  using namespace Jacobi;

  vt::initialize(argc, argv);
  
  JacobiConfig app_cfg(argc, argv);

  Kokkos::initialize(argc, argv);
  {
  
  ResilienceConfig res_cfg(argc, argv);
  
  const int this_node = vt::theContext()->getNode();
 

  
  // Object group of all nodes that take part in computation
  // Used to determine whether the computation is finished
  auto grp_proxy = vt::theObjGroup()->makeCollective<Detector>("notify");
  //Register on all nodes or just one. Registering on all = a few more init messages.
  res_cfg->register_to(MAIN_LOOP, grp_proxy);

  // Create the decomposition into objects
  auto col_proxy = vt::makeCollection<Solver>("jacobi")
        .bounds(app_cfg.colRange)
        .bulkInsert()
        .wait();
  res_cfg->register_to(MAIN_LOOP, col_proxy);

  

  //Initialize application, unless recovering.
  int recover_iter = res_cfg->latest_version(MAIN_LOOP);
  bool recovering = recover_iter >= 1;
  if(recovering && this_node == 0) fmt::print("Recovering to iteration {}\n", recover_iter);
  if(!recovering) {
    auto cfg_copy = app_cfg;
    cfg_copy.objProxy = grp_proxy;
    vt::runInEpochCollective([&]{
      col_proxy.broadcastCollective<&Solver::init>(cfg_copy);
    });
  }


  size_t iter_count = recovering ? recover_iter : 1;
  //Accounting for count being 1-based.
  const size_t max_iter = app_cfg.maxIter + 1;

  //Manage app/phase boundaries accoring to resilience configs.
  //Phase boundaries supercede and restart app boundary.
  size_t next_phase_boundary = iter_count + res_cfg.iters_per_phase;
  if(res_cfg.iters_per_phase == -1) next_phase_boundary = max_iter+1;

  size_t next_epoch_boundary = iter_count + res_cfg.iters_per_epoch;
  if(res_cfg.iters_per_epoch == -1) next_epoch_boundary = max_iter+1;

double loop_begin_s = vt::timing::getCurrentTime();
  //Iteratively solve
  while (!isWorkDone(grp_proxy) && iter_count < max_iter) {
    const size_t next_boundary = std::min(std::min(next_phase_boundary, next_epoch_boundary), max_iter);

    if(this_node == 0) fmt::print(stderr, "Launching iterations [{},{}]\n", iter_count, next_boundary-1);

    vt::runInEpochCollective(fmt::format("Jacobi iters [{}-{}]", iter_count, next_boundary-1), [&]{
      while( !isWorkDone(grp_proxy) && iter_count < next_boundary ){
        res_cfg->run(MAIN_LOOP, iter_count, [&]{
          if(this_node == 0){
            res_cfg->register_to_active(col_proxy);
            res_cfg->register_to_active(grp_proxy);
          }
          
          col_proxy.broadcastCollective<&Solver::iterate>(next_boundary-1);
          iter_count++;
        
        }, res_cfg.checkpoint_filter);
      }
    });

    vt::runInEpochCollective(fmt::format("Jacobi reduce {}", iter_count), [&]{
      col_proxy.broadcastCollective<&Solver::reduce>();
    });


    //Update boundaries as necessary.
    if(iter_count == next_phase_boundary) {
      next_phase_boundary = iter_count + res_cfg.iters_per_phase;
      vt::thePhase()->nextPhaseCollective();
    }
    if(iter_count == next_epoch_boundary) {
      next_epoch_boundary = iter_count + res_cfg.iters_per_epoch;
    }
  }

double loop_end_s = vt::timing::getCurrentTime();
if(this_node == 0) fmt::print("Loop took {}s\n", loop_end_s - loop_begin_s);


  vt::runInEpochCollective("Cleanup objects", [&]{
    vt::theCollection()->destroy(col_proxy);
    vt::theObjGroup()->destroyCollective(grp_proxy);
    res_cfg.reset();
  });
  //Done! Free the resilient context for cleanup.
  }
  Kokkos::finalize();
  
  vt::finalize();
}


ResilienceConfig::ResilienceConfig(int argc, char** argv){
  //Where to find .json configuration file for KokkosResilience.
  //  --config <filename>
  //NOTE: safest by far to give an absolute path (as well as within the config file)
  std::string config_filename = "config_jacobi.json";

  //Which context to use for consistency management
  //  --mode <VT/MPI>
  std::string context_mode = "VT";

  //How often to checkpoint, in iterations. Two special cases:
  //negative: Do not checkpoint
  //       0: Use the default checkpoint filter, settable within the config file
  //  --freq <int>
  int checkpoint_frequency = 0;

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
    //Let VTContext manage epoch boundaries (apart from phases)
    if(iters_per_epoch == 0) iters_per_epoch = -1;
    context = kr::make_context(vt::theContext(), config_filename);
  } else if(context_mode == "MPI"){
    //App must put boundary before checkpoint, as MPIContext is unaware of VT tasks
    if(iters_per_epoch == 0) iters_per_epoch = checkpoint_frequency;
    context = kr::make_context(MPI_COMM_WORLD, config_filename);
  } else throw std::invalid_argument("Valid modes are [VT, MPI]. Provided mode: " + context_mode);
 
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
  
  //Must be positive, or -1 for no phase/epoch bounding
  assert(iters_per_phase > 0 || iters_per_phase == -1);
  assert(iters_per_epoch > 0 || iters_per_epoch == -1);

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
