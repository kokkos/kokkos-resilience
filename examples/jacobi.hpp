#include <vt/transport.h>
#include <Kokkos_Core.hpp>

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

  bool isWorkFinished(){
    return finished; 
  }

  void workFinished(){
    finished = true; 
  }
};
using DetectorProxy = vt::objgroup::proxy::Proxy<Detector>;

bool isWorkDone( DetectorProxy const& proxy) {
  return proxy.get()->isWorkFinished();
};

struct Config {
  vt::Index3D colRange;
  vt::Index3D dataRange;

  double tolerance = 1e-2;
  int maxIter = 100;

  bool asyncCheckpoint = false;
  bool debug = false;

  DetectorProxy objProxy;

  Config() = default;

  template<typename SerT>
  void serialize(SerT& s){
    s | colRange | dataRange | tolerance | maxIter | asyncCheckpoint | objProxy | debug;
  }

  Config(int argc, char** argv){
    int numXObjs = 4;
    int numYObjs = 4;
    int numZObjs = 4;

    int numXElms = 50;
    int numYElms = 50;
    int numZElms = 50;

    for(int i = 0; i < argc; i++){
      std::string arg = argv[i];
      //Set input decomposition (i.e. dimenions of col_proxy)
      if(       arg == "--decomp"){
        numXObjs = std::stoi(argv[++i]);
        numYObjs = std::stoi(argv[++i]);
        numZObjs = std::stoi(argv[++i]);

      //Set input size per element of col_proxy
      //Currently just uses x and makes a cube.
      } else if(arg == "--input"){
        numXElms = std::stoi(argv[++i]);
        numYElms = std::stoi(argv[++i]);
        numZElms = std::stoi(argv[++i]);

      } else if(arg == "--max-iters") {
        maxIter = std::stoi(argv[++i]);

      } else if(arg == "--tolerance") {
        tolerance = std::stod(argv[++i]);
      } else if(arg == "--async-serialize") {
        asyncCheckpoint = true;
      } else if(arg == "--jacobi-debug"){
        debug = true;
      }
    }

    //using BaseIndexType = typename vt::Index3D::DenseIndexType;
    colRange = vt::Index3D(numXObjs, numYObjs, numZObjs);
    dataRange = vt::Index3D(numXElms, numYElms, numZElms);

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
};

struct Solver : vt::Collection<Solver,vt::Index3D> {
  Config cfg;

  //Lower for edges
  int nNeighbors = 6;
  
  std::array<size_t, 3> nElms;

  //Previous iteration's data, current result, input right hand side.
  Kokkos::View<double***> previous, result, rhs;

  int iter = 0, nextLaunchIter = 0, reduceIter = -1;
  std::deque<vt::EpochType> epochQueue;

  int nRecv = 0, nEarlyRecv = 0;

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

        //if(vt::theContext()->getNode() == 0) fmt::print(stderr, "{}@{} delaying on epoch {} @{}\n", getIndex().toString(), iter, epochQueue.back(), vt::timing::getCurrentTime());
        chkpt_epoch = vt::theTerm()->makeEpochRooted(fmt::format("{} checkpointing!\n", getIndex().toString()));
        vt::theTerm()->addDependency(last_iter_epoch, chkpt_epoch);
        epochQueue.push_back(chkpt_epoch);

        kr::Util::VT::delaySerializeUntil(last_iter_epoch);
      }
    }

    if(!s.isSizing() && iter%150 != 0) fmt::print(stderr, "Warning: {} checkpointing iter {}\n", getIndex().toString(), iter);
    //auto begin_time = vt::timing::getCurrentTime();
    //if(!s.isSizing() && vt::theContext()->getNode() == 0) fmt::print("{}@{} beginning serialization @{}\n", getIndex().toString(), iter, begin_time);
    
    vt::trace::TraceScopedNote trace_obj(
      fmt::format("{} {}@{}", s.isSizing()?"Sizing":"Serializing", getIndex().toString(), iter),
      kr::Context::VT::VTContext::serialize_proxy
    );

    vt::Collection<Solver,vt::Index3D>::serialize(s);
    s | cfg | nNeighbors | nElms | previous | result | rhs | iter | nextLaunchIter | reduceIter;
    trace_obj.end();
    
    if(chkpt_epoch != vt::no_epoch) {
      epochQueue.pop_front();
      vt::theTerm()->finishedEpoch(chkpt_epoch);
    }
    //auto end_time = vt::timing::getCurrentTime();
    //if(!s.isSizing() && vt::theContext()->getNode() == 0) fmt::print("{}@{} finished serialization in {}s @{}\n", getIndex().toString(), iter, end_time-begin_time, end_time);
  }

  
  void init(Config cfg_){
    cfg = cfg_;

    auto idx = getIndex();
    for(int dim = 0; dim < 3; dim++){
      nElms[dim] = cfg.dataRange[dim]/cfg.colRange[dim];
      if(idx[dim] < (cfg.dataRange[dim]%cfg.colRange[dim]))
        nElms[dim]++;
      if(nElms[dim] <= 1){
        fmt::print(stderr, "{} running with only {} elements in dimension {}\n", getIndex().toString(), nElms[dim], dim);
        assert(nElms[dim] > 0);
      }
      
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

  //Wait for any queued iterations to finish, then iterate.
  void iterate(int in_reduceIter){
    reduceIter = in_reduceIter;
    vt::EpochType iterEpoch = vt::theTerm()->makeEpochRooted(
        fmt::format("{} iteration {}", getIndex().toString(), nextLaunchIter)
    );
    
    vt::EpochType predEpoch = epochQueue.empty() ? vt::no_epoch : epochQueue.back();
    if(predEpoch != vt::no_epoch) {
      vt::theTerm()->addDependency(predEpoch, iterEpoch);
    }
    
    epochQueue.push_back(iterEpoch);

    int launchIter = ++nextLaunchIter;

    if(predEpoch == vt::no_epoch){
      _iterate(launchIter);
    } else {
if(cfg.debug){
      fmt::print("{}@{}, delaying iterate@{} until previous finished. Predecessor epoch: {}, iter epoch: {}\n", 
                          getIndex().toString(), iter, launchIter, predEpoch, iterEpoch);
}
      vt::EpochType parent_epoch = vt::theTerm()->getEpoch();
      vt::theTerm()->addLocalDependency(parent_epoch);

      vt::theTerm()->addAction(predEpoch, [launchIter, this, parent_epoch]{
        vt::theTerm()->pushEpoch(parent_epoch);
        getCollectionProxy()[getIndex()].send<&Solver::_iterate>(launchIter);
        vt::theTerm()->popEpoch(parent_epoch);
        
        vt::theTerm()->releaseLocalDependency(parent_epoch);
      });
    }
  }

  //Iterate once.
  void _iterate(int target) {
if(target != iter+1) fmt::print(stderr, "{}@{} expected target iteration {}\n", getIndex().toString(), iter, target-1);
    assert(target == iter+1);
    
    //Early recvs are just recvs now
    nRecv = nEarlyRecv;
    bool already_recvd = nRecv == nNeighbors;
    nEarlyRecv = 0;
    iter++;

    
    //Swap previous and result, will overwrite result w/ new
    auto tmp = result;
    result = previous;
    previous = tmp;
    
    //Send edge values to neighbors
    auto proxy = getCollectionProxy();
    auto idx = getIndex();
    
    //vt::theMsg()->pushEpoch(epochQueue.front());
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
    //vt::theMsg()->popEpoch(epochQueue.front());

    if(already_recvd) compute();
  };

  //Reduce the global error
  void reduce() {
/*    while(iter != reduceIter){
      assert(!epochQueue.empty());
if(cfg.debug){
      fmt::print("{}@{}, delaying reduce until reaching reduceIter of {}\n", 
                  getIndex().toString(), iter, reduceIter);
}
      //Retry until whatever chunk of queued up iterations finishes.
      vt::theTerm()->addDependency(epochQueue.back(), vt::theTerm()->getEpoch());
      vt::theTerm()->addAction(epochQueue.back(), [&]{
        reduce();
      });
      return;
    }*/

    using ValT = typename Kokkos::MinMax<double>::value_type;
    ValT minMax;

    Kokkos::parallel_reduce(Range({1,1,1}, {nElms[0]-1, nElms[1]-1, nElms[2]-1}),
      KOKKOS_LAMBDA (const int x, const int y, const int z, ValT& l_minMax){
        auto& val = result(x,y,z);
        l_minMax.min_val = std::min(l_minMax.min_val, val);
        l_minMax.max_val = std::max(l_minMax.max_val, val);
      }, Kokkos::MinMax<double>(minMax));
    
    double max = std::max(minMax.min_val*-1, minMax.max_val);
   
if(cfg.debug){
    fmt::print("{}@{}, local max: {}\n", getIndex().toString(), iter, max);
}
    
    auto proxy = getCollectionProxy();
    proxy.reduce<&Solver::checkCompleted, vt::collective::MaxOp>(proxy(0,0,0), max);
  };
  
  void checkCompleted(double maxNorm) {
    bool within_tolerance = maxNorm < cfg.tolerance;
    bool timed_out = iter == cfg.maxIter;
    bool done = within_tolerance || timed_out;

    if(done){
      if(within_tolerance)
        fmt::print("\n # Jacobi reached tolerance threshold ({}<{}) in {} iterations\n\n", maxNorm, cfg.tolerance, iter);
      else if(timed_out)
        fmt::print("\n # Jacobi reached maximum iterations ({}) with while above tolerance ({}>{})\n\n", iter, maxNorm, cfg.tolerance);
      cfg.objProxy.broadcast<&Detector::workFinished>();
    } else {
      fmt::print(" # Iteration {} reached with maxNorm {}\n", iter, maxNorm);
    }
  };

  void exchange(vt::Index3D sender, Kokkos::View<double**, Kokkos::LayoutStride> ghost, int in_iter) {
    bool early = in_iter != iter;
    if(early) assert(in_iter == iter+1);

    vt::Index3D dir = getIndex() - sender;
    auto dest = getGhostPlane(early ? result : previous, dir);
    Kokkos::deep_copy(dest, ghost);

    if(early) nEarlyRecv++;
    else nRecv++;
    
if(cfg.debug){
    fmt::print("{}: Received from {} for iter {}. My iter: {}. Early? {}. Recvs: {}/{}, w/ {} early.\n", 
        getIndex().toString(), ghost.extent(0), ghost.extent(1), sender.toString(), in_iter, iter, early, nRecv+1, 
        nNeighbors, nEarlyRecv);
}

    if(!early && nRecv == nNeighbors){
      auto iter_epoch = epochQueue.front();
      vt::theTerm()->pushEpoch(iter_epoch);
      getCollectionProxy()[getIndex()].send<&Solver::compute>();
      vt::theTerm()->popEpoch(iter_epoch);
      vt::theTerm()->finishedEpoch(iter_epoch);
    }
  };
  
private:
  void compute() {
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
   
    //if(iter%150 == 0 && vt::theContext()->getNode() == 0) fmt::print(stderr, "{}@{} finished @{}\n", getIndex().toString(), iter, vt::timing::getCurrentTime());
    //Now finalize this iteration.
    assert(!epochQueue.empty());
    epochQueue.pop_front();
  };

  
  Kokkos::View<double**, Kokkos::LayoutStride>
  getGhostPlane(Kokkos::View<double***> in, vt::Index3D dir){
    return getPlane(in, dir, true);
  }
  Kokkos::View<double**, Kokkos::LayoutStride>
  getPlane(Kokkos::View<double***> in, vt::Index3D dir, bool ghost = false){
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
};
using SolverProxy = vt::vrt::collection::CollectionProxy<Solver, vt::Index3D>;

}
