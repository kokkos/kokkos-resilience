/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

// Header guard, format: directory_directory_filename
#ifndef INC_RESILIENCE_OPENMP_RESOPENMP_HPP
#define INC_RESILIENCE_OPENMP_RESOPENMP_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_OPENMP ) 

#include <Kokkos_Core_fwd.hpp>

#include <cstddef>
#include <iosfwd>

#include <vector>

#include <Kokkos_ScratchSpace.hpp> 
#include <Kokkos_Parallel.hpp> 
#include <Kokkos_TaskScheduler.hpp> 
#include <Kokkos_Layout.hpp> 
#include <impl/Kokkos_Tags.hpp> 

//This space specific
#include <Kokkos_Serial.hpp>
#include <Kokkos_OpenMP.hpp> //Kokkos-fork 
#include <Kokkos_HostSpace.hpp> //Kokkos-fork 
#include "ResHostSpace.hpp" //Resilient

/*------------------------------------------------------------------------*/

namespace KokkosResilience {

// Resilient Kokkos Execution Space that uses OpenMP to run multiple
// duplicates of a desired kernel, each on a separate partition.
class ResOpenMP : public Kokkos::OpenMP {
  public:

    //Type declarations for execution spaces following API
    typedef ResOpenMP				     execution_space; //Tag class as Kokkos execution space
    typedef ResHostSpace		  	        memory_space; //Preferred memory space
    typedef Kokkos::Device<execution_space,memory_space> device_type; //Preferred device_type
    typedef memory_space::size_type   			   size_type; //Preferred size_type
    typedef Kokkos::LayoutRight				array_layout; //Preferred array layout
    typedef Kokkos::ScratchMemorySpace<OpenMP>  scratch_memory_space; //Preferred scratch memory
  
/*------------------------------------*/

    //Do not delete constructor, defined in cpp
  
    ~ResOpenMP() {}
    ResOpenMP();
  
    ResOpenMP( ResOpenMP && ) = default ;
    ResOpenMP( const ResOpenMP & ) = default ;
    ResOpenMP & operator = ( ResOpenMP && ) = default ;
    ResOpenMP & operator = ( const ResOpenMP & ) = default ;

/*------------------------------------*/


    // Print configuration information to the given output stream.
    static void print_configuration(std::ostream&, const bool verbose = false);

    // The instance running a parallel algorithm
    inline static bool in_parallel(OpenMP const& = OpenMP()) noexcept;

    // Wait until all dispatched functors complete on the given instance
    // This is a no-op on OpenMP
    static void impl_static_fence(OpenMP const& = OpenMP()) noexcept;

    void fence();

    // Does the given instance return immediately after launching
    // a parallel algorithm
    // This always returns false on OpenMP
    inline static bool is_asynchronous(OpenMP const& = OpenMP()) noexcept;

    // Partition the default instance into new instances without creating new masters
    // This is a no-op on OpenMP since the default instance cannot be partitioned
    // without promoting other threads to 'master'
    static std::vector<OpenMP> partition(...);

    // Non-default instances should be ref-counted so that when the last
    // is destroyed the instance resources are released
    // This is a no-op on OpenMP since a non default instance cannot be created
    static OpenMP create_instance(...);
  
    // Partition the default instance and call 'f' on each new 'master' thread
    // Func is a functor with the following signature
    // void( int partition_id, int num_partitions )
    template <typename F>
    static void partition_master(F const& f, int requested_num_partitions = 0,
         			 int requested_partition_size = 0);

    // use UniqueToken
    static int concurrency();

    /*****************************************
    // MAY NEED TO INSERT DEPRECATED CODE HERE
    *****************************************/
    
    static void impl_initialize(int thread_count = -1);

    // The default execution space initialized for current 'master' thread
    static bool impl_is_initialized() noexcept;

    // Free any resources being consumed by the default execution space
    static void impl_finalize();

    inline static int impl_thread_pool_size() noexcept;
 
    // The rank of the executing thread in this thread pool
    KOKKOS_INLINE_FUNCTION static int impl_thread_pool_rank() noexcept;

    inline static int impl_thread_pool_size(int depth);

    // use UniqueToken
    inline static int impl_max_hardware_threads() noexcept;

    // use UniqueToken
    KOKKOS_INLINE_FUNCTION static int impl_hardware_thread_id() noexcept;

    static int impl_get_current_max_threads() noexcept;

    static const char* name();
    uint32_t impl_instance_id() const noexcept { return 0; }

}; //template class ResOpenMP execution space

} //namespace KokkosResilience

/*------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

// Type trait that holds space accessibility information
// Derived from accessibility matrix (more cases in Kokkos API wiki)
// Specialized to ResHostSpace, may need to extend to more
template <>
struct MemorySpaceAccess
  < KokkosResilience::ResHostSpace
  , Kokkos::OpenMP::scratch_memory_space
  > 
{
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = false };
};

/*------------------------------------------------------------------------*/

// Specialized to ResHostSpace, extend to more
template <>
struct VerifyExecutionCanAccessMemorySpace
  < KokkosResilience::ResHostSpace
  , KokkosResilience::ResOpenMP::scratch_memory_space
  > 
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

}  // namespace Impl
}  // namespace Kokkos

/*------------------------------------------------------------------------*/

// This is the specialization which corrects the profiling error
// In the future, it should be corrected by a new implementation in main Kokkos
// which is more general. For now this will suffice.

namespace Kokkos {
namespace Tools { 
namespace Experimental { 
template <> 
struct DeviceTypeTraits<KokkosResilience::ResOpenMP> { 
  static constexpr DeviceType id = DeviceType::OpenMP; 
}; 
}  // namespace Experimental 
}  // namespace Tools 
}  // namespace Kokkos

/*------------------------------------------------------------------------*/

#include <OpenMP/Kokkos_OpenMP_Exec.hpp>
#include <OpenMP/Kokkos_OpenMP_Team.hpp>
#include "OpenMPResParallel.hpp" // Resilient specific parallel functors
#include <OpenMP/Kokkos_OpenMP_Task.hpp>

//#include <KokkosExp_MDRangePolicy.hpp>

/*------------------------------------------------------------------------*/

#endif //#if defined( KOKKOS_ENABLE_OPENMP )
#endif //INC_RESILIENCE_OPENMP_RESOPENMP_HPP
