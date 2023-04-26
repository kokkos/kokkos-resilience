/*
 *
 *                        Kokkos v. 3.0
 *       Copyright (2020) National Technology & Engineering
 *               Solutions of Sandia, LLC (NTESS).
 *
 * Under the terms of Contract DE-NA0003525 with NTESS,
 * the U.S. Government retains certain rights in this software.
 *
 * Kokkos is licensed under 3-clause BSD terms of use:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the Corporation nor the names of the
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Questions? Contact Christian R. Trott (crtrott@sandia.gov)
 */

#ifndef RES_CUDA_SPACE_HPP
#define RES_CUDA_SPACE_HPP

#include <Kokkos_Macros.hpp>
#if defined(KR_ENABLE_CUDA)

#include <Kokkos_Core.hpp> //to fix header inclusion errors later

//#include "ResCuda.hpp"

#include <iosfwd>
#include <typeinfo>
#include <string>
#include <memory>

#include <typeinfo>
#include <map>


/*--------------------------------------------------------------------------*/

#ifdef KOKKOS_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
extern "C" bool kokkos_impl_cuda_pin_uvm_to_host();
extern "C" void kokkos_impl_cuda_set_pin_uvm_to_host(bool);
#endif

/*--------------------------------------------------------------------------*/
/*
namespace Kokkos {
namespace Impl {

//TODO::Need this for deepcopy memspace now?
template <>
struct is_cuda_type_space<KokkosResilience::ResCudaSpace> : public std::false_type {};

}
}*/

namespace KokkosResilience {

//forward declaration of ResCuda for temlating DeepCopy on MemSpace
class ResCuda;

class ResCudaSpace : public Kokkos::CudaSpace {
 public:

  // Type declarations for execution space
  using base_space      = Kokkos::CudaSpace;    // Parent space derived from
  using memory_space    = ResCudaSpace; // Tag class as Kokkos memory space
  using size_type       = unsigned int; // Preferred size (note: not size_t)
  using execution_space = ResCuda; // Preferred execution space
  using resilient_space = ResCudaSpace; // resilient tag
  using device_type     = Kokkos::Device<execution_space, memory_space>;

  // Parent class constructors
  using Kokkos::CudaSpace::CudaSpace;

 private:

  friend class Kokkos::Impl::SharedAllocationRecord< ResCudaSpace, void >; 

}; // class ResCudaSpace

} //namespace Kokkos Resilience


/*--------------------------------------------------------------------------*/

namespace Kokkos {

namespace Impl {

/*--------------------------------------------------------------------------*/

//TODO: UVM Epansion

template <>
struct MemorySpaceAccess< KokkosResilience::ResCudaSpace, KokkosResilience::ResCudaSpace > {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy   = true };
};

template <>
struct MemorySpaceAccess< KokkosResilience::ResCudaSpace, Kokkos::HostSpace > {
  enum { assignable = false };
  enum { accessible = false };
  enum { deepcopy   = true };
};

template <>
struct MemorySpaceAccess< Kokkos::HostSpace, KokkosResilience::ResCudaSpace > {
  enum { assignable = false };
  enum { accessible = false };
  enum { deepcopy   = true };
};

template <>
struct MemorySpaceAccess< KokkosResilience::ResCudaSpace, Kokkos::CudaSpace > {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy   = true };
};

template <>
struct MemorySpaceAccess< Kokkos::CudaSpace, KokkosResilience::ResCudaSpace > {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy   = true };
};

/*
template <>
struct MemorySpaceAccess <KokkosResilience::ResCudaSpace, KokkosResilience::ResCudaSpace>
     : MemorySpaceAccess <Kokkos::CudaSpace, Kokkos::CudaSpace>{
};

template < typename OtherSpace >
struct MemorySpaceAccess <KokkosResilience::ResCudaSpace, OtherSpace
                         , std::enable_if_t< !std::is_same<OtherSpace, Kokkos::AnonymousSpace>::value >> 
     : MemorySpaceAccess <Kokkos::CudaSpace, OtherSpace>{
};

template < typename OtherSpace >
struct MemorySpaceAccess <OtherSpace, KokkosResilience::ResCudaSpace
                         , std::enable_if_t< !std::is_same<OtherSpace, Kokkos::AnonymousSpace>::value >> 
     : MemorySpaceAccess <OtherSpace, Kokkos::CudaSpace>{
};

static_assert(Kokkos::Impl::MemorySpaceAccess<KokkosResilience::ResCudaSpace,
                                              KokkosResilience::ResCudaSpace>::assignable, "");
*/
} // namespace Impl
} // namespace Kokkos


/*--------------------------------------------------------------------------*/

// TODO: Ask about templating in Kokkos cuda space, design of ResCudaSpace??
// Implications for design of ResHostSpace?

namespace Kokkos {
namespace Impl {

//! Specialize this trait to control the behavior of deep_copy in CudaSpace
template <>
struct ExecutionSpaceAlias< KokkosResilience::ResCuda >{

  using type = Cuda;

};

//TODO: OLD PARADIGM KEPT, BUT NOW THEORETICALLY NOT DUMPED INTO NON-CUDA DEEPCOPIES

/// default case
template <class MemSpace>
struct is_res_cuda_type_space : std::false_type {};

// specialization for ResCudaSpace
template <>
struct is_res_cuda_type_space<KokkosResilience::ResCudaSpace> : std::true_type {};


//TODO::Need this for deepcopy memspace now?
template <>
struct is_cuda_type_space<KokkosResilience::ResCudaSpace> : public std::true_type {};


// DeepCopy (dest, src)

// Template deep copy: ResCudaSpace -> Host
//
// last template parameter of inherited is deduced
// MemSpace::base_space must be defined (in order to point to the correct deepcopy attached
// to the correct CudaSpace)
template <class MemSpace>
struct DeepCopy< Kokkos::HostSpace, MemSpace, KokkosResilience::ResCuda, std::enable_if_t<is_res_cuda_type_space<MemSpace>::value>>
     : DeepCopy< Kokkos::HostSpace, typename MemSpace::base_space, Cuda >
{
  using DeepCopy< Kokkos::HostSpace, typename MemSpace::base_space, Cuda >::DeepCopy;
};

// Template deep copy: Host -> ResCudaSpace
template <class MemSpace>
struct DeepCopy< MemSpace, Kokkos::HostSpace, KokkosResilience::ResCuda, std::enable_if_t<is_res_cuda_type_space<MemSpace>::value>>
     : DeepCopy< typename MemSpace::base_space, Kokkos::HostSpace, Cuda >
{
  using DeepCopy< typename MemSpace::base_space, Kokkos::HostSpace, Cuda >::DeepCopy;
};

// Template deep copy: ResCuda -> ResCuda (only, because base_space must be defined)
template <class MemSpace1, class MemSpace2>
struct DeepCopy<MemSpace1, MemSpace2, KokkosResilience::ResCuda, std::enable_if_t<is_res_cuda_type_space<MemSpace1>::value && is_res_cuda_type_space<MemSpace2>::value>>
    : DeepCopy< typename MemSpace1::base_space, typename MemSpace2::base_space, Cuda>
{
  using DeepCopy< typename MemSpace1::base_space, typename MemSpace2::base_space, Cuda>::DeepCopy;
};

} // namespace Impl
} // namespace Kokkos

/*--------------------------------------------------------------------------*/

namespace Kokkos {

namespace Impl {

template <>
class SharedAllocationRecord< KokkosResilience::ResCudaSpace , void >
  : public SharedAllocationRecord< Kokkos::CudaSpace , void >
{
private:
  friend class SharedAllocationRecord< Kokkos::CudaSpace , void >;
  // Cuda specific
  friend class HostInaccessibleSharedAllocationRecordCommon<Kokkos::CudaSpace>;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete;

  using RecordBase = SharedAllocationRecord< void , void >;
  // Cuda specific
  using base_t = HostInaccessibleSharedAllocationRecordCommon<Kokkos::CudaSpace>;

 protected:
//  ~SharedAllocationRecord(); // ... one more time about inheriting destructors fine point?
  SharedAllocationRecord() = default;

// NOTE! This is based on Damien's fix in OMP, note in Cuda space the execution space differences
  template <class ExecutionSpace>
  SharedAllocationRecord(
      const ExecutionSpace                              & exec_space,
      const KokkosResilience::ResCudaSpace               & arg_space,
      const std::string                                  & arg_label,
      const size_t                                    arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &base_t::deallocate)
      : SharedAllocationRecord <Kokkos::CudaSpace, void> (exec_space, arg_space, arg_label, arg_alloc_size, arg_dealloc) {}
   
  SharedAllocationRecord(
      const KokkosResilience::ResCudaSpace               & arg_space,
      const std::string                                  & arg_label,
      const size_t                                    arg_alloc_size,
      const RecordBase::function_type    arg_dealloc = &base_t::deallocate)
  :SharedAllocationRecord< Kokkos::CudaSpace, void >(arg_space,arg_label,arg_alloc_size,arg_dealloc){}

}; // class SharedAllocationRecord



} // namespace Impl
} // namespace Kokkos

/*--------------------------------------------------------------------------*/

#endif // #if defined( KR_ENABLE_CUDA )
#endif // #define RES_CUDA_SPACE_HPP

