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

#include "config.hpp"
#include "solver.hpp"

//Label for the resilience region.
const std::string MAIN_LOOP = "Jacobi main loop";

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
  vt::initialize(argc, argv);
  const int this_node = vt::theContext()->getNode();
  
  Jacobi::Config app_cfg(argc, argv);
  ResilienceConfig res_cfg(argc, argv, app_cfg);

  int recover_iter = res_cfg->latest_version(MAIN_LOOP);
  bool recovering = recover_iter >= 1;
  if(recovering && this_node == 0) 
    fmt::print("Recovering to iteration {}\n", recover_iter);
  
  
  //Object group of all nodes that take part in computation
  // Used to determine whether the computation is finished
  auto grp_proxy = vt::theObjGroup()->makeCollective<Jacobi::Detector>("notify");

  //Collection of Solver objects that perform the work.
  auto col_proxy = vt::makeCollection<Jacobi::Solver>("jacobi")
        .bounds(app_cfg.colRange).bulkInsert().wait();
  if(!recovering) {
    vt::runInEpochCollective([=]{
      col_proxy.broadcastCollective<&Jacobi::Solver::init>(app_cfg, grp_proxy);
    });
  }
 
  //Register our objects, labels are pulled from the VT labels
  res_cfg->register_to(MAIN_LOOP, grp_proxy);
  res_cfg->register_to(MAIN_LOOP, col_proxy);


  size_t iter = 1;
  if(recovering) iter = recover_iter;

  const size_t max_iter = app_cfg.maxIter + 1; //Our iter is 1-based

  size_t next_phase_boundary = res_cfg.iters_per_phase + iter;
  size_t next_epoch_boundary = res_cfg.iters_per_epoch + iter;
  size_t next_boundary = std::min(std::min(next_phase_boundary, next_epoch_boundary), max_iter);

  while (!isWorkDone(grp_proxy) && iter < max_iter) {
    vt::runInEpochCollective(fmt::format("Jacobi iters [{}-{}]", iter, next_boundary-1), [&]{
      if(this_node == 0) fmt::print(stderr, "Running iterations [{},{}]\n", iter, next_boundary-1);

      for( ; iter < next_boundary && !isWorkDone(grp_proxy); iter++){
        res_cfg->run(MAIN_LOOP, iter, [&]{
          res_cfg->register_to_active(col_proxy);
          res_cfg->register_to_active(grp_proxy);
          
          col_proxy.broadcastCollective<&Jacobi::Solver::iterate>();
        }, res_cfg.checkpoint_filter);
      }
    });

    vt::runInEpochCollective(fmt::format("Jacobi reduce {}", iter), [&]{
      col_proxy.broadcastCollective<&Jacobi::Solver::reduce>();
    });

    //Update boundaries as necessary.
    if(iter == next_epoch_boundary) next_epoch_boundary += res_cfg.iters_per_epoch;
    if(iter == next_phase_boundary) {
      next_phase_boundary += res_cfg.iters_per_phase;
      vt::thePhase()->nextPhaseCollective();
    }
    next_boundary = std::min(std::min(next_phase_boundary, next_epoch_boundary), max_iter);
  }
  
  vt::finalize();
  }
  Kokkos::finalize();
}


