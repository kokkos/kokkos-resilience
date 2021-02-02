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

#include <impl/TrackDuplicates.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_CudaSpace.hpp>
#include <typeinfo>
#include <map>

/*--------------------------------------------------------------------------*/

namespace KokkosResilience {

// Resilient HostSpace

// It should function the same as regular host space, except only for
// OpenMP at the moment. This can be added to at a later time. Inherits 
// from HostSpace.
class ResHostSpace : public Kokkos::HostSpace {
  public:
  
    // Type declarations for execution spaces following API
    typedef ResHostSpace	memory_space; // Tag class as Kokkos memory space
    typedef size_t 		size_type; // Preferred size type
    typedef Kokkos::OpenMP 	execution_space; // Preferred execution space

    // Every memory space has a default execution space.  This is
    // useful for things like initializing a View (which happens in
    // parallel using the View's default execution space).

    /*--------------------------------------------------------------
    IF DEFINE MACRO SWITCH HERE WHEN EXPANDING TO MORE MEMORY SPACES  
    --------------------------------------------------------------*/

    typedef Kokkos::Device<execution_space, memory_space> device_type; // Preferred device type

    // Default memory space instance
    ResHostSpace();
    ResHostSpace( ResHostSpace && rhs )      = default;
    ResHostSpace( const ResHostSpace & rhs ) = default;
    ResHostSpace & operator = ( ResHostSpace && ) = default;
    ResHostSpace & operator = ( const ResHostSpace & ) = default;
    ~ResHostSpace()                           = default;
   
    static void clear_duplicates_list();

    static std::map<std::string, KokkosResilience::DuplicateTracker * > duplicate_map;

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

} // namespace Impl

} // namespace Kokkos

/*--------------------------------------------------------------------------*/

namespace Kokkos {

namespace Impl {

template <>
class SharedAllocationRecord< KokkosResilience::ResHostSpace , void >
  : public SharedAllocationRecord< void , void > 
{
private:
  friend SharedAllocationRecord< Kokkos::HostSpace , void >;

  typedef SharedAllocationRecord< void, void> RecordBase;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete;

  static void deallocate( RecordBase * );

#ifdef KOKKOS_DEBUG
  // Root record for tracked allocations from this ResHostSpace instance
  static RecordBase s_root_record;
#endif

  const KokkosResilience::ResHostSpace m_space;

protected:
  ~SharedAllocationRecord()

#if defined( \
    KOKKOS_IMPL_INTEL_WORKAROUND_NOEXCEPT_SPECIFICATION_VIRTUAL_FUNCTION)
      noexcept
#endif
      ;
  SharedAllocationRecord() = default;

  SharedAllocationRecord(
      const KokkosResilience::ResHostSpace               & arg_space, 
      const std::string                                  & arg_label,
      const size_t                                    arg_alloc_size,
      const RecordBase::function_type    arg_dealloc = & deallocate);

public:
  inline std::string get_label() const {
    return std::string(RecordBase::head()->m_label);
  }

  KOKKOS_INLINE_FUNCTION static SharedAllocationRecord* allocate(
      const KokkosResilience::ResHostSpace& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size) {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
#else
    (void)arg_space;
    (void)arg_label;
    (void)arg_alloc_size;
    return (SharedAllocationRecord*)0;
#endif
  }

  // Allocate tracked memory in the space
  static void* allocate_tracked(const KokkosResilience::ResHostSpace& arg_space,
                                 const std::string& arg_label,
                                 const size_t arg_alloc_size);

  // Reallocate tracked memory in the space
  static void* reallocate_tracked(void* const arg_alloc_ptr,
                                  const size_t arg_alloc_size);

  // Deallocate tracked memory in the space
  static void deallocate_tracked(void* const arg_alloc_ptr);

  static SharedAllocationRecord* get_record(void* arg_alloc_ptr);

  static void print_records(std::ostream&, const KokkosResilience::ResHostSpace&,
                            bool detail = false);

}; // class SharedAllocationRecord

}  // namespace Impl

}  // namespace Kokkos

/*--------------------------------------------------------------------------*/

namespace KokkosResilience{
  
template <class Type>
class SpecDuplicateTracker<Type, Kokkos::OpenMP> : public DuplicateTracker {
  public:
    typedef typename std::remove_reference<Type>::type nr_type;
    typedef typename std::remove_pointer<nr_type>::type np_type;
    typedef typename std::remove_extent<np_type>::type ne_type;
    typedef typename std::remove_const<ne_type>::type rd_type;
    typedef CombineFunctor<rd_type, Kokkos::OpenMP> comb_type;

    comb_type m_cf;

    inline SpecDuplicateTracker() : DuplicateTracker(), m_cf() {}

    inline SpecDuplicateTracker(const SpecDuplicateTracker& rhs)
        : DuplicateTracker(rhs), m_cf(rhs.m_cf) {}

     virtual bool combine_dups();
     virtual void set_func_ptr();
};

template <class Type>
void SpecDuplicateTracker<Type, Kokkos::OpenMP>::set_func_ptr() {}

template <class Type>
bool SpecDuplicateTracker<Type, Kokkos::OpenMP>::combine_dups() {
  
  bool success;
  bool trigger = 1;

  if (dup_cnt != 3) {
    printf("must have 3 duplicates !!!\n"); 
    return 0;
  }
  int N = data_len / sizeof(rd_type);
  m_cf.load_ptrs( static_cast<rd_type*>(original_data)
                , static_cast<rd_type*>(dup_list[0])
                , static_cast<rd_type*>(dup_list[1])
                , static_cast<rd_type*>(dup_list[2]), N );

  comb_type local_cf(m_cf);
  
  Kokkos::parallel_for( N, KOKKOS_LAMBDA(int i) {
    success = local_cf.exec(i);
    if (!success) trigger = 0;
  });

  return trigger;
 
}

} //namespace KokkosResilience

/*--------------------------------------------------------------------------*/

#endif // #define INC_RESILIENCE_OPENMP_RESHOSTSPACE_HPP
