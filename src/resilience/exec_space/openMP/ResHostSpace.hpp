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

#ifndef INC_RESILIENCE_OPENMP_RESHOSTSPACE_HPP
#define INC_RESILIENCE_OPENMP_RESHOSTSPACE_HPP

#include <cstring>
#include <string>
#include <iosfwd>
#include <typeinfo>

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>

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
    using memory_space    = ResHostSpace;   // Tag class as Kokkos memory space
    using size_type       = size_t;         // Preferred size type
    using execution_space = Kokkos::OpenMP; // Preferred execution space
    using resilient_space = ResHostSpace;

    // Every memory space has a default execution space.  This is
    // useful for things like initializing a View (which happens in
    // parallel using the View's default execution space).

    using device_type = Kokkos::Device<execution_space, memory_space>; // Preferred device type

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

/*--------------------------------------*/

} // namespace Impl

} // namespace Kokkos

/*--------------------------------------------------------------------------*/

namespace Kokkos {

namespace Impl {

// Template deep copy: ResHost -> ResHost
template <class ExecutionSpace>
struct DeepCopy< KokkosResilience::ResHostSpace, KokkosResilience::ResHostSpace, ExecutionSpace>
     : DeepCopy< Kokkos::HostSpace, Kokkos::HostSpace, ExecutionSpace>
{
  using DeepCopy< Kokkos::HostSpace, Kokkos::HostSpace, ExecutionSpace>::DeepCopy;
};

// Template deep copy: Host -> ResHost
template <class ExecutionSpace>
struct DeepCopy< Kokkos::HostSpace, KokkosResilience::ResHostSpace, ExecutionSpace>
     : DeepCopy< Kokkos::HostSpace, Kokkos::HostSpace, ExecutionSpace>
{
  using DeepCopy< Kokkos::HostSpace, Kokkos::HostSpace, ExecutionSpace>::DeepCopy;
};

// Template deep copy: ResHost -> Host
// Absolutely essential for ViewHooks wrapper
template <class ExecutionSpace>
struct DeepCopy< KokkosResilience::ResHostSpace, Kokkos::HostSpace, ExecutionSpace>
     : DeepCopy< Kokkos::HostSpace, Kokkos::HostSpace, ExecutionSpace>
{
  using DeepCopy< Kokkos::HostSpace, Kokkos::HostSpace, ExecutionSpace>::DeepCopy;
};

} // namespace Impl
} // namespace Kokkos

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

template <class ExecutionSpace>
SharedAllocationRecord(
      const ExecutionSpace                              & exec_space,
      const KokkosResilience::ResHostSpace               & arg_space,
      const std::string                                  & arg_label,
      const size_t                                    arg_alloc_size,
      const RecordBase::function_type    arg_dealloc = & deallocate)
  :SharedAllocationRecord< Kokkos::HostSpace, void >(exec_space, arg_space,arg_label,arg_alloc_size,arg_dealloc){}

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
