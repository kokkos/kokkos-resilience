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
#ifndef INC_RESILIENCE_OPENMP_RESHOSTSPACE_HPP
#define INC_RESILIENCE_OPENMP_RESHOSTSPACE_HPP

#include <cstring>
#include <string>
#include <iosfwd>
#include <typeinfo>

/*--------------------------------
*** MAY NEED MORE HEADERS HERE**
--------------------------------*/
#include <Kokkos_Core_fwd.hpp>

// Resilience 
#include <Kokkos_Macros.hpp>
// Not including KOKKOS_ENABLE_OPENMP because it's host space, may
// desire it to work with other spaces later.

//#include <impl/TrackDuplicates.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_CudaSpace.hpp>
#include <typeinfo>
#include <map>

//Fix this if included
//#include <openmp/OpenMPResSubscriber.hpp>

/*--------------------------------------------------------------------------*/

namespace KokkosResilience {

// Resilient HostSpace

// It should function the same as regular host space, except only for
// OpenMP at the moment. This can be added to at a later time. Inherits 
// from HostSpace.
class ResHostSpace : public Kokkos::HostSpace {
  public:

    // Type declarations for execution spaces following API
    typedef ResHostSpace        memory_space; // Tag class as Kokkos memory space
    typedef size_t              size_type; // Preferred size type
    typedef Kokkos::OpenMP      execution_space; // Preferred execution space
    using   resilient_space =   ResHostSpace;

    // Every memory space has a default execution space.  This is
    // useful for things like initializing a View (which happens in
    // parallel using the View's default execution space).
    /*--------------------------------------------------------------
    TODO: IF DEFINE MACRO SWITCH HERE WHEN EXPANDING TO MORE MEMORY SPACES  
    --------------------------------------------------------------*/

    typedef Kokkos::Device<execution_space, memory_space> device_type; // Preferred device type

    // Use parent class constructors
    using Kokkos::HostSpace::HostSpace;

  private:

    friend class Kokkos::Impl::SharedAllocationRecord< ResHostSpace , void >;

}; // class ResHostSpace

}  // namespace KokkosResilience


/*--------------------------------------------------------------------------*/

namespace Kokkos {

namespace Impl {

static_assert(Kokkos::Impl::MemorySpaceAccess<KokkosResilience::ResHostSpace,
                                              KokkosResilience::ResHostSpace>::assignable, "");

/*--------------------------------------*/

// Memory Space Access specializations, from view accessiblity matrix 

template <>
struct MemorySpaceAccess< KokkosResilience::ResHostSpace, Kokkos::HostSpace > {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy   = true };
};

template <>
struct MemorySpaceAccess< Kokkos::HostSpace, KokkosResilience::ResHostSpace > {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy   = true };
};

/*--------------------------------------
NEED MORE CASES TO BE COMPLETELY GUARDED
--------------------------------------*/

/*--------------------------------------*/

} // namespace Impl

} // namespace Kokkos

/*--------------------------------------------------------------------------*/

/*---------------------------------------
MAY NEED HOST MIRROR SPECIALIZATIONS HERE
---------------------------------------*/

/*--------------------------------------------------------------------------*/

namespace Kokkos {

namespace Impl {

// Template deep copy: ResHost -> ResHost 
template <class ExecutionSpace>
struct DeepCopy< KokkosResilience::ResHostSpace, KokkosResilience::ResHostSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    hostspace_parallel_deepcopy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    hostspace_parallel_deepcopy(dst, src, n);
    exec.fence();
  }
};

// Template deep copy: Host -> ResHost
// Absolutely essential for ViewHooks wrapper 
template <class ExecutionSpace>
struct DeepCopy< Kokkos::HostSpace, KokkosResilience::ResHostSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    hostspace_parallel_deepcopy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    hostspace_parallel_deepcopy(dst, src, n);
    exec.fence();
  }
};

// Template deep copy: ResHost -> Host
// Absolutely essential for ViewHooks wrapper
template <class ExecutionSpace>
struct DeepCopy< KokkosResilience::ResHostSpace, Kokkos::HostSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    hostspace_parallel_deepcopy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    hostspace_parallel_deepcopy(dst, src, n);
    exec.fence();
  }
};

/*--------------------------------------
NEED MORE CASES TO BE COMPLETELY GUARDED
--------------------------------------*/


} // namespace Impl
} // namespace Kokkos

/*--------------------------------------------------------------------------*/

namespace Kokkos {

namespace Impl {

/*--------------------------------------
NEED MORE CASES TO BE COMPLETELY GUARDED
--------------------------------------*/
/*
#if defined ( KOKKOS_ENABLE_CUDA )
// Running in ResHostSpace, attempting to access CudaSpace
template<>
struct VerifyExecutionCanAccessMemorySpace< KokkosResilience::ResHostSpace , Kokkos::CudaSpace >
{
  enum { value = false };
  inline static void verify( void ) { KokkosResilience::ResHostSpace::access_error(); }
  inline static void verify( const void * p ) { KokkosResilience::ResHostSpace::access_error(p); }
};
#endif

// Running in ResHostSpace and attempting to access an unknown space: throw error
template< class OtherSpace >
struct VerifyExecutionCanAccessMemorySpace<
  typename std::enable_if< ! std::is_same<KokkosResilience::ResHostSpace,OtherSpace>::value , KokkosResilience::ResHostSpace >::type ,
  OtherSpace >
{
  enum { value = false };
  inline static void verify( void )
    { Kokkos::abort("resilient OpenMP code attempted to access unknown space memory"); }
  inline static void verify( const void * )
    { Kokkos::abort("resilient OpenMP code attempted to access unknown space memory"); }
};
*/
} // namespace Impl

} // namespace Kokkos

/*--------------------------------------------------------------------------*/

namespace Kokkos {

namespace Impl {

template <>
class SharedAllocationRecord< KokkosResilience::ResHostSpace , void >
  : public SharedAllocationRecord< HostSpace , void >
{
private:
  friend SharedAllocationRecord< Kokkos::HostSpace , void >;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete;

  using RecordBase = SharedAllocationRecord< void , void >;

 protected:

  SharedAllocationRecord() = default;

  SharedAllocationRecord(
      const KokkosResilience::ResHostSpace               & arg_space,
      const std::string                                  & arg_label,
      const size_t                                    arg_alloc_size,
      const RecordBase::function_type    arg_dealloc = & deallocate)
  :SharedAllocationRecord< Kokkos::HostSpace, void >(arg_space,arg_label,arg_alloc_size,arg_dealloc){}

}; // class SharedAllocationRecord

}  // namespace Impl

}  // namespace Kokkos

/*--------------------------------------------------------------------------*/

#endif // #define INC_RESILIENCE_OPENMP_RESHOSTSPACE_HPP
