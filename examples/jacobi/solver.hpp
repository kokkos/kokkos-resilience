#ifndef JACOBI_SOLVER_HPP
#define JACOBI_SOLVER_HPP

#include <vt/transport.h>
#include <Kokkos_Core.hpp>
#include "config.hpp"

//
// This code applies a few steps of the Jacobi iteration to
// the linear system  A x = 0
// where is a banded symmetric positive definite matrix.
// The initial guess for x is a made-up non-zero vector.
// The exact solution is the vector 0.
//
// The matrix A is square and invertible.
// The number of rows is ((number of objects) * (number of rows per object))
//
// Such a matrix A is obtained when using 2nd-order finite difference
// for discretizing
//
// -d^2 u / dx^2 -d^2 u / dy^2 - -d^2 u / dz^2  = f   on  [0, 1] x [0, 1] x [0, 1]
//
// with homogeneous Dirichlet condition
//
// u = 0 on the boundary of [0, 1] x [0, 1] x [0, 1]
//
// using a uniform grid with grid size
//
// 1 / ((number of objects) * (number of rows per object) + 1)
//

namespace Jacobi {

struct Detector {
  bool finished = false;

  template <typename Serializer>
  void serialize(Serializer& s) {
    s | finished;
  }

  bool isWorkFinished();
  void workFinished();
};
using DetectorProxy = vt::objgroup::proxy::Proxy<Detector>;

bool isWorkDone( DetectorProxy const& proxy);

struct Solver : vt::Collection<Solver,vt::Index3D> {
  Config cfg;
  DetectorProxy detector;

  //Lower for edges
  int nNeighbors = 6;
  
  std::array<size_t, 3> nElms;

  //Previous iteration's data, current result, input right hand side.
  Kokkos::View<double***> previous, result, rhs;

  //iter tracks the iteration locally completed or currently in progress.
  //nextLaunchIter is just used for sanity checking.
  int iter = 0, nextLaunchIter = 0;
  std::deque<vt::EpochType> epochQueue;

  //Count ghost messages received. We might get some "early" messages for our next iteration
  int nRecv = 0, nEarlyRecv = 0;

  //3D range Kokkos execution policy
  using Range = Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::IndexType<int>>;

public:
  Solver() = default;

  template <typename Serializer>
  void serialize(Serializer& s) {
    vt::EpochType chkpt_epoch = vt::no_epoch;
    if(s.hasTraits(vt::vrt::CheckpointInternalTrait()) && cfg.asyncCheckpoint && !s.isSizing()){
      //Checkpointing waits for enqueued iterations to finish
      if(!epochQueue.empty()) {
        vt::EpochType last_iter_epoch = epochQueue.back();

        chkpt_epoch = vt::theTerm()->makeEpochRooted(fmt::format("{} checkpointing!\n", getIndex().toString()));
        vt::theTerm()->addDependency(last_iter_epoch, chkpt_epoch);
        epochQueue.push_back(chkpt_epoch);

        kr::Util::VT::delaySerializeUntil(last_iter_epoch);
      }
    }

    vt::trace::TraceScopedNote trace_obj(
      fmt::format("{} {}@{}", s.isSizing()?"Sizing":"Serializing", getIndex().toString(), iter),
      kr::Context::VT::VTContext::serialize_proxy
    );

    vt::Collection<Solver,vt::Index3D>::serialize(s);
    s | cfg | detector | nNeighbors | nElms | previous | result | rhs | iter;
    trace_obj.end();
    
    if(chkpt_epoch != vt::no_epoch) {
      epochQueue.pop_front();
      vt::theTerm()->finishedEpoch(chkpt_epoch);
    }
  }

  
  void init(Config cfg_, DetectorProxy detector_);

  //Requests another iteration be launched. 
  //Manages waiting on any outstanding iterations to finish.
  void iterate();

  //Internal. Perform the actual iteration steps.
  void _iterate();

  //Reduce the global error. Not currently asynchronously safe.
  void reduce();
 
  //Internal. Gets reduced global error and notifies of completion if finished.
  void checkCompleted(double maxNorm);

  //Internal. Handles incoming ghost values.
  void exchange(vt::Index3D sender, Kokkos::View<double**, Kokkos::LayoutStride> ghost, int in_iter);
  
private:
  void compute();
 
  //Get a Kokkos subview of the edge plane of input view.
  //  Which edge to get is defined by dir, which should have a single non-zero dimension
  //  Choose low or high edge of that dimension with a -1 or 1 value.
  //(Essentially, dir=neighborIndex-myIndex gives you the ghost plane in the direction of neighbor)
  Kokkos::View<double**, Kokkos::LayoutStride>
  getGhostPlane(Kokkos::View<double***> in, vt::Index3D dir);

  //As getGhostPlane, but by default gets the edge plane of local data not the ghost values.
  Kokkos::View<double**, Kokkos::LayoutStride>
  getPlane(Kokkos::View<double***> in, vt::Index3D dir, bool ghost = false);
};

using SolverProxy = vt::vrt::collection::CollectionProxy<Solver, vt::Index3D>;

}

#endif
