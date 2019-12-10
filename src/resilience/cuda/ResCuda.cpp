#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_CUDA

#include "ResCuda.hpp"

#include <Cuda/Kokkos_Cuda_Instance.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>

#include <cstdlib>
#include <vector>
#include <string>

namespace KokkosResilience {

ResCuda::size_type ResCuda::detect_device_count()
{ return Cuda::detect_device_count();}

int ResCuda::concurrency()
{ return Cuda::concurrency();}

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
   int ResCuda::is_initialized()
   { return Kokkos::Impl::CudaInternal::singleton().is_initialized(); }

   void ResCuda::finalize()
   {
      Kokkos::Impl::CudaInternal::singleton().finalize();

      #if defined(KOKKOS_ENABLE_PROFILING)
        Kokkos::Profiling::finalize();
      #endif
   }

   void ResCuda::initialize( const Cuda::SelectDevice config , size_t num_instances )
   {
     Kokkos::Impl::CudaInternal::singleton().initialize( config , 0 );

      #if defined(KOKKOS_ENABLE_PROFILING)
        Kokkos::Profiling::initialize();
      #endif
   }
#else
  //! Has been initialized
  int ResCuda::impl_is_initialized() {
     return Kokkos::Impl::CudaInternal::singleton().is_initialized();
  }

  void ResCuda::impl_finalize() {
      Kokkos::Impl::CudaInternal::singleton().finalize();

      #if defined(KOKKOS_ENABLE_PROFILING)
        Kokkos::Profiling::finalize();
      #endif
  }

  void ResCuda::impl_initialize( const SelectDevice config, const size_t num_instances ) {
     Kokkos::Impl::CudaInternal::singleton().initialize( config.cuda_device_id , 0 );

      #if defined(KOKKOS_ENABLE_PROFILING)
        Kokkos::Profiling::initialize();
      #endif
  }
#endif


std::vector<unsigned>
ResCuda::detect_device_arch()
{
   return Cuda::detect_device_arch() ;
}

ResCuda::size_type ResCuda::device_arch()
{
  return Cuda::device_arch() ;
}


ResCuda::ResCuda()
  : Cuda() {
}

ResCuda::ResCuda( cudaStream_t stream )
  : Cuda( stream )
{}

void ResCuda::print_configuration( std::ostream & s , const bool )
{ Kokkos::Impl::CudaInternal::singleton().print_configuration( s ); }

void ResCuda::fence()
{
  Cuda::fence();
}

const char* ResCuda::name() { return "ResCuda"; }

} // namespace KokkosResilience

#else

void KOKKOS_CORE_SRC_CUDA_IMPL_PREVENT_LINK_ERROR() {}

#endif // KOKKOS_ENABLE_CUDA

