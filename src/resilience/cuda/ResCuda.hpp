#ifndef INC_RESILIENCE_CUDA_RESCUDA_HPP
#define INC_RESILIENCE_CUDA_RESCUDA_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_CUDA )

#include <Kokkos_Core_fwd.hpp>

#include <iosfwd>
#include <vector>

#include <Kokkos_Cuda.hpp>
#include <Kokkos_CudaSpace.hpp>
#include "ResCudaSpace.hpp"

#include <Kokkos_Parallel.hpp>
#include <Kokkos_TaskScheduler.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <impl/Kokkos_Tags.hpp>


/*--------------------------------------------------------------------------*/


/*--------------------------------------------------------------------------*/

namespace KokkosResilience {


/// \class ResCuda
/// \brief Kokkos Execution Space that uses CUDA to run on GPUs.
///
/// An "execution space" represents a parallel execution model.  It tells Kokkos
/// how to parallelize the execution of kernels in a parallel_for or
/// parallel_reduce.  For example, the Threads execution space uses Pthreads or
/// C++11 threads on a CPU, the OpenMP execution space uses the OpenMP language
/// extensions, and the Serial execution space executes "parallel" kernels
/// sequentially.  The ResCuda execution space uses NVIDIA's CUDA programming
/// model to execute kernels in parallel on GPUs.
class ResCuda : Kokkos::Cuda {
public:
  //! \name Type declarations that all Kokkos execution spaces must provide.
  //@{

  //! Tag this class as a kokkos execution space
  typedef ResCuda                  execution_space ;

#if defined( KOKKOS_ENABLE_CUDA_UVM )
  //! This execution space's preferred memory space.
  typedef ResCudaUVMSpace          memory_space ;
#else
  //! This execution space's preferred memory space.
  typedef ResCudaSpace             memory_space ;
#endif

  //! This execution space preferred device_type
  typedef Kokkos::Device<execution_space,memory_space> device_type;

  //! The size_type best suited for this execution space.
  typedef Kokkos::Cuda::size_type  size_type ;

  //! This execution space's preferred array layout.
  typedef Kokkos::LayoutLeft            array_layout ;

  //!
  typedef Kokkos::ScratchMemorySpace< Cuda >  scratch_memory_space ;

  //@}
  //--------------------------------------------------
  //! \name Functions that all Kokkos devices must implement.
  //@{

  /// \brief True if and only if this method is being called in a
  ///   thread-parallel function.
  KOKKOS_INLINE_FUNCTION static int in_parallel() {
#if defined( __CUDA_ARCH__ )
    return true;
#else
    return false;
#endif
  }


  /// \brief Wait until all dispatched functors complete.
  ///
  /// The parallel_for or parallel_reduce dispatch of a functor may
  /// return asynchronously, before the functor completes.  This
  /// method does not return until all dispatched functors on this
  /// device have completed.
  void fence();

  /** \brief  Return the maximum amount of concurrency.  */
  static int concurrency();

  //! Print configuration information to the given output stream.
  static void print_configuration( std::ostream & , const bool detail = false );

  //@}
  //--------------------------------------------------
  //! \name  ResCuda space instances

  ~ResCuda() {}
  ResCuda();
  explicit ResCuda( cudaStream_t stream );

  ResCuda( ResCuda && ) = default ;
  ResCuda( const ResCuda & ) = default ;
  ResCuda & operator = ( ResCuda && ) = default ;
  ResCuda & operator = ( const ResCuda & ) = default ;

  //--------------------------------------------------------------------------
  //! \name Device-specific functions
  //@{


#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
  //! Free any resources being consumed by the device.
  static void finalize();

  //! Has been initialized
  static int is_initialized();

  //! Initialize, telling the CUDA run-time library which device to use.
  static void initialize( const SelectDevice = SelectDevice()
                        , const size_t num_instances = 1 );
#else
  //! Free any resources being consumed by the device.
  static void impl_finalize();

  //! Has been initialized
  static int impl_is_initialized();

  //! Initialize, telling the CUDA run-time library which device to use.
  static void impl_initialize( const SelectDevice = SelectDevice()
                        , const size_t num_instances = 1 );
#endif

  /// \brief ResCuda device architecture of the selected device.
  ///
  /// This matches the __CUDA_ARCH__ specification.
  static size_type device_arch();

  //! Query device count.
  static size_type detect_device_count();

  /** \brief  Detect the available devices and their architecture
   *          as defined by the __CUDA_ARCH__ specification.
   */
  static std::vector<unsigned> detect_device_arch();

  cudaStream_t cuda_stream() const { return Cuda::cuda_stream() ; }
  int          cuda_device() const { return Cuda::cuda_device() ; }

  //@}
  //--------------------------------------------------------------------------

  static const char* name();

};

} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

template<>
struct MemorySpaceAccess
  < KokkosResilience::ResCudaSpace
  , KokkosResilience::ResCuda::scratch_memory_space
  >
{
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy   = false };
};

#if defined( KOKKOS_ENABLE_CUDA_UVM )

// If forcing use of UVM everywhere
// then must assume that ResCudaUVMSpace
// can be a stand-in for ResCudaSpace.
// This will fail when a strange host-side execution space
// that defines ResCudaUVMSpace as its preferredmemory space.

template<>
struct MemorySpaceAccess
  < KokkosResilience::ResCudaUVMSpace
  , KokkosResilience::ResCuda::scratch_memory_space
  >
{
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy   = false };
};

#endif


template<>
struct VerifyExecutionCanAccessMemorySpace
  < KokkosResilience::ResCudaSpace
  , KokkosResilience::ResCuda::scratch_memory_space
  >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};


} // namespace Impl
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

#include <Cuda/Kokkos_Cuda_KernelLaunch.hpp>
#include <Cuda/Kokkos_Cuda_View.hpp>
#include <Cuda/Kokkos_Cuda_Team.hpp>
#include "CudaResParallel.hpp"
#include <Cuda/Kokkos_Cuda_Task.hpp>
#include <Cuda/Kokkos_Cuda_UniqueToken.hpp>

//#include <KokkosExp_MDRangePolicy.hpp>
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_CUDA ) */
#endif  // INC_RESILIENCE_CUDA_RESCUDA_HPP

