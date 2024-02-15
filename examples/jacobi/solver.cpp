#include "solver.hpp"
namespace Jacobi {
bool Detector::isWorkFinished(){return finished;}
void Detector::workFinished(){finished = true;}
bool isWorkDone( DetectorProxy const& proxy) {
  return proxy.get()->isWorkFinished();
};

void Solver::init(Config cfg_, DetectorProxy detector_){
  cfg = cfg_;
  detector = detector_;

  auto idx = getIndex();
  for(int dim = 0; dim < 3; dim++){
    nElms[dim] = cfg.dataRange[dim]/cfg.colRange[dim];
    if(idx[dim] < (cfg.dataRange[dim]%cfg.colRange[dim]))
      nElms[dim]++;
    assert(nElms[dim] > 0);

    nElms[dim] += 2; 
    if(idx[dim] == 0) nNeighbors--;
    if(idx[dim] == cfg.colRange[dim]-1) nNeighbors--;
  }
 

  //previous/result views are swapped, so keep labels generic.
  previous = Kokkos::View<double***>("ViewA", nElms[0], nElms[1], nElms[2]);
  result   = Kokkos::View<double***>("ViewB", nElms[0], nElms[1], nElms[2]);
  rhs      = Kokkos::View<double***>("RHS",   nElms[0], nElms[1], nElms[2]);
  Kokkos::deep_copy(previous, 0);
  Kokkos::deep_copy(result, 0);
  Kokkos::deep_copy(rhs, 0);
  

  //
  // Set the initial vector to the values of
  // a "high-frequency" function
  //
  double hx = 1.0 / (cfg.dataRange.x()+1);
  double hy = 1.0 / (cfg.dataRange.y()+1);
  double hz = 1.0 / (cfg.dataRange.z()+1);

  int maxDim = std::max(std::max(cfg.dataRange[0], cfg.dataRange[1]), cfg.dataRange[2]);
  int nf = 3 * int(maxDim+1) / 6;

  std::array<size_t, 3> offsets;
  for(int dim = 0; dim < 3; dim++){
    offsets[dim] = idx[dim] * (cfg.dataRange[dim]/cfg.colRange[dim]);
    offsets[dim] += std::min(idx[dim], (cfg.dataRange[dim]%cfg.colRange[dim]));
  }

  Kokkos::parallel_for(Range({1,1,1}, {nElms[0]-1, nElms[1]-1, nElms[2]-1}), 
    KOKKOS_LAMBDA (const int x, const int y, const int z){
      double val = pow((offsets[0]+x)*hx, 2);
      val += pow((offsets[1]+y)*hy, 2);
      val += pow((offsets[2]+z)*hz, 2);

      result(x,y,z) = sin(nf * M_PI * val);
  });
}


void Solver::iterate(){
  //Grab the last iteration's epoch
  vt::EpochType predEpoch = epochQueue.empty() ? vt::no_epoch : epochQueue.back();

  //Make an epoch for this iteration, and add to the queue.
  vt::EpochType iterEpoch = vt::theTerm()->makeEpochRooted(
      fmt::format("{} iteration {}", getIndex().toString(), nextLaunchIter)
  );
  epochQueue.push_back(iterEpoch);

  if(predEpoch == vt::no_epoch){
    //Just launch up the next iteration
    _iterate();
  } else {
    //Wait until prior iteration finishes to launch this one.

    //Grab up current epoch so this iteration's messages are correctly assigned
    vt::EpochType parentEpoch = vt::theTerm()->getEpoch();
    vt::theTerm()->addLocalDependency(parentEpoch);

    //Add an action to run once prior iteration finishes
    vt::theTerm()->addAction(predEpoch, [this, parentEpoch]{
      vt::theTerm()->pushEpoch(parentEpoch);
      getCollectionProxy()[getIndex()].send<&Solver::_iterate>();
      vt::theTerm()->popEpoch(parentEpoch);
      
      vt::theTerm()->releaseLocalDependency(parentEpoch);
    });
    
    //Make the epoch dependency explicit, so VT can reduce termination detection messages.
    vt::theTerm()->addDependency(predEpoch, iterEpoch);
  }
}

void Solver::_iterate() {
  iter++;
  
  //Early recvs are just recvs now
  nRecv = nEarlyRecv;
  nEarlyRecv = 0;
  
  //Swap previous and result, will overwrite result w/ new
  std::swap(result, previous);
 
  //Send ghost values to neighbors
  auto proxy = getCollectionProxy();
  auto idx = getIndex();
  
  for(int dim = 0; dim < 3; dim++){
    std::array<int, 3> dir = {0,0,0};
    
    if(idx[dim] > 0){
      dir[dim] = -1;
      proxy[idx + dir].send<&Solver::exchange>(idx, getPlane(previous, dir), iter);
    }
    if(idx[dim] < cfg.colRange[dim]-1){
      dir[dim] = 1;
      proxy[idx + dir].send<&Solver::exchange>(idx, getPlane(previous, dir), iter);
    }
  }

  //We may have gotten all our ghost values while still working on last iteration
  if(nRecv == nNeighbors) compute();
};
  
void Solver::reduce() {
  using ValT = typename Kokkos::MinMax<double>::value_type;
  ValT minMax;

  Kokkos::parallel_reduce(Range({1,1,1}, {nElms[0]-1, nElms[1]-1, nElms[2]-1}),
    KOKKOS_LAMBDA (const int x, const int y, const int z, ValT& l_minMax){
      auto& val = result(x,y,z);
      l_minMax.min_val = std::min(l_minMax.min_val, val);
      l_minMax.max_val = std::max(l_minMax.max_val, val);
    }, Kokkos::MinMax<double>(minMax));
  
  double max = std::max(minMax.min_val*-1, minMax.max_val);
 
  auto proxy = getCollectionProxy();
  proxy.reduce<&Solver::checkCompleted, vt::collective::MaxOp>(proxy(0,0,0), max);
};

void Solver::checkCompleted(double maxNorm) {
  bool within_tolerance = maxNorm < cfg.tolerance;
  bool timed_out = iter == cfg.maxIter;
  bool done = within_tolerance || timed_out;

  if(done){
    if(within_tolerance)
      fmt::print("\n # Jacobi reached tolerance threshold ({}<{}) in {} iterations\n\n", maxNorm, cfg.tolerance, iter);
    else if(timed_out)
      fmt::print("\n # Jacobi reached maximum iterations ({}) while above tolerance ({}>{})\n\n", iter, maxNorm, cfg.tolerance);
    detector.broadcast<&Detector::workFinished>();
  } else {
    fmt::print(" # Iteration {} reached with maxNorm {}\n", iter, maxNorm);
  }
};

void Solver::exchange(vt::Index3D sender, Kokkos::View<double**, Kokkos::LayoutStride> ghost, int in_iter) {
  bool early = in_iter != iter;
  if(early) assert(in_iter == iter+1);

  vt::Index3D dir = sender - getIndex();
  auto dest = getGhostPlane(early ? result : previous, dir);
  Kokkos::deep_copy(dest, ghost);

  if(early) nEarlyRecv++;
  else nRecv++;
  
  if(!early && nRecv == nNeighbors){
    auto iter_epoch = epochQueue.front();
    
    vt::theTerm()->pushEpoch(iter_epoch);
    getCollectionProxy()[getIndex()].send<&Solver::compute>();
    vt::theTerm()->popEpoch(iter_epoch);

    vt::theTerm()->finishedEpoch(iter_epoch);
  }
}

void Solver::compute() {
  //
  //---- Jacobi iteration step for
  //---- A banded matrix for the 8-point stencil
  //---- [ 0.0  -1.0   0.0]
  //---- [-1.0]  
  //---- [-1.0   6.0  -1.0]  
  //---- [-1.0]  
  //---- [ 0.0  -1.0   0.0]
  //---- rhs_ right hand side vector
  //
  Kokkos::parallel_for(Range({1,1,1}, {nElms[0]-1, nElms[1]-1, nElms[2]-1}), 
    KOKKOS_LAMBDA (const int x, const int y, const int z){
      result(x,y,z) = (1.0/6.0) * (
          rhs(x,y,x) + previous(x-1,y,z) + previous(x+1,y,z) + 
          previous(x,y-1,z) + previous(x,y+1,z) + previous(x,y,z-1) +
          previous(x,y,z+1));
  });
 
  //No longer waiting on this iteration
  assert(!epochQueue.empty());
  epochQueue.pop_front();
};
  
Kokkos::View<double**, Kokkos::LayoutStride>
Solver::getGhostPlane(Kokkos::View<double***> in, vt::Index3D dir){
  return getPlane(in, dir, true);
}

Kokkos::View<double**, Kokkos::LayoutStride>
Solver::getPlane(Kokkos::View<double***> in, vt::Index3D dir, bool ghost){
  using Dir = vt::Index3D;
  if(dir == Dir(-1,0,0))
    return Kokkos::subview(in, ghost ? 0 : 1, Kokkos::ALL(), Kokkos::ALL());
  if(dir == Dir(1,0,0))
    return Kokkos::subview(in, ghost ? nElms[0]-1 : nElms[0]-2, Kokkos::ALL(), Kokkos::ALL()); 
  if(dir == Dir(0,-1,0))
    return Kokkos::subview(in, Kokkos::ALL(), ghost ? 0 : 1, Kokkos::ALL());
  if(dir == Dir(0,1,0))
    return Kokkos::subview(in, Kokkos::ALL(), ghost ? nElms[0]-1 : nElms[0]-2, Kokkos::ALL());
  if(dir == Dir(0,0,-1))
    return Kokkos::subview(in, Kokkos::ALL(), Kokkos::ALL(), ghost ? 0 : 1);
  if(dir == Dir(0,0,1))
    return Kokkos::subview(in, Kokkos::ALL(), Kokkos::ALL(), ghost ? nElms[0]-1 : nElms[0]-2);

  assert(false);
  return Kokkos::subview(in, 0, Kokkos::ALL(), Kokkos::ALL());
}
}
