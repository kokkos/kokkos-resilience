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

#include <cstdio>
#include <algorithm>
#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_MemorySpace.hpp>

//#if defined(KOKKOS_ENABLE_PROFILING)
//#include <impl/Kokkos_Profiling_Interface.hpp>
//#endif

/*--------------------------------------------------------
IF SWITCHES FOR NON-PREFERRED MEMORY HERE, WHEN EXPANDING
---------------------------------------------------------*/

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include <iostream>
#include <sstream>

// Resilience headers
#include "../Resilience.hpp"
#include "ResHostSpace.hpp"

#include <impl/Kokkos_Error.hpp>
#include <Kokkos_Atomic.hpp>

#if (defined(KOKKOS_ENABLE_ASM) || defined(KOKKOS_ENABLE_TM)) && \
    defined(KOKKOS_ENABLE_ISA_X86_64) && !defined(KOKKOS_COMPILER_PGI)
#include <immintrin.h>
#endif

/*--------------------------------------------------------------------------*/

// Duplicate tracker cleanup

namespace KokkosResilience {

ResHostSpace::ResHostSpace()
  : HostSpace() {}

} // namespace KokkosResilience

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_ENABLE_DEBUG
SharedAllocationRecord<void, void>
    SharedAllocationRecord<KokkosResilience::ResHostSpace, void>::s_root_record;
#endif

void SharedAllocationRecord<KokkosResilience::ResHostSpace , void>::deallocate(
  SharedAllocationRecord< void , void> * arg_rec)
  {
    delete static_cast<SharedAllocationRecord *>(arg_rec);
  }

SharedAllocationRecord<KokkosResilience::ResHostSpace, void>::~SharedAllocationRecord()
#if defined( \
    KOKKOS_IMPL_INTEL_WORKAROUND_NOEXCEPT_SPECIFICATION_VIRTUAL_FUNCTION)
    noexcept
#endif
{
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::deallocateData(
        Kokkos::Profiling::SpaceHandle(KokkosResilience::ResHostSpace::name()),
        RecordBase::m_alloc_ptr->m_label, data(), size());
  }
#endif

  m_space.deallocate(SharedAllocationRecord<void, void>::m_alloc_ptr,
                     SharedAllocationRecord<void, void>::m_alloc_size);
}

SharedAllocationHeader *_do_allocation(KokkosResilience::ResHostSpace const &space,
                                       std::string const &label,
                                       size_t alloc_size) {
  try {
    return reinterpret_cast<SharedAllocationHeader *>(
        space.allocate(alloc_size));
  } catch (Experimental::RawMemoryAllocationFailure const &failure) {
    if (failure.failure_mode() == Experimental::RawMemoryAllocationFailure::
                                      FailureMode::AllocationNotAligned) {
      // TODO: delete the misaligned memory
    }

    std::cerr << "Kokkos failed to allocate memory for label \"" << label
              << "\".  Allocation using MemorySpace named \"" << space.name()
              << " failed with the following error:  ";
    failure.print_error_message(std::cerr);
    std::cerr.flush();
    Kokkos::Impl::throw_runtime_exception("Memory allocation failure");
  }
  return nullptr;  // unreachable
}


SharedAllocationRecord<KokkosResilience::ResHostSpace, void>::SharedAllocationRecord(
    const KokkosResilience::ResHostSpace &arg_space, const std::string &arg_label,
    const size_t arg_alloc_size,
    const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function

    /* TODO: REMOVE TEMP CALLSIGN
     *   SharedAllocationRecord(
#ifdef KOKKOS_ENABLE_DEBUG
      SharedAllocationRecord* arg_root,
#endif
      SharedAllocationHeader* arg_alloc_ptr, size_t arg_alloc_size,
      function_type arg_dealloc);*/

    : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_ENABLE_DEBUG
          &SharedAllocationRecord<KokkosResilience::ResHostSpace, void>::s_root_record,
#endif
          Impl::checked_allocation_with_header(arg_space, arg_label,
                                               arg_alloc_size),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc),
      m_space(arg_space) {
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(
        Kokkos::Profiling::SpaceHandle(arg_space.name()), arg_label, data(),
        arg_alloc_size);
  }
#endif
  // Fill in the Header information
  RecordBase::m_alloc_ptr->m_record =
      static_cast<SharedAllocationRecord<void, void> *>(this);

  strncpy(RecordBase::m_alloc_ptr->m_label, arg_label.c_str(),
          SharedAllocationHeader::maximum_label_length);
  // Set last element zero, in case c_str is too long
  RecordBase::m_alloc_ptr
      ->m_label[SharedAllocationHeader::maximum_label_length - 1] = (char)0;
}

/*--------------------------------------------------------------------------*/

void *SharedAllocationRecord<KokkosResilience::ResHostSpace, void>::allocate_tracked(
    const KokkosResilience::ResHostSpace &arg_space, const std::string &arg_alloc_label,
    const size_t arg_alloc_size) {
  if (!arg_alloc_size) return nullptr;

  SharedAllocationRecord *const r =
      allocate(arg_space, arg_alloc_label, arg_alloc_size);

  RecordBase::increment(r);

  return r->data();
}

void SharedAllocationRecord<KokkosResilience::ResHostSpace, void>::deallocate_tracked(
    void *const arg_alloc_ptr) {
  if (arg_alloc_ptr != nullptr) {
    SharedAllocationRecord *const r = get_record(arg_alloc_ptr);

    RecordBase::decrement(r);
  }
}

void *SharedAllocationRecord<KokkosResilience::ResHostSpace, void>::reallocate_tracked(
    void *const arg_alloc_ptr, const size_t arg_alloc_size) {
  SharedAllocationRecord *const r_old = get_record(arg_alloc_ptr);
  SharedAllocationRecord *const r_new =
      allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

  Kokkos::Impl::DeepCopy<KokkosResilience::ResHostSpace, KokkosResilience::ResHostSpace>(
      r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size()));

  RecordBase::increment(r_new);
  RecordBase::decrement(r_old);

  return r_new->data();
}

SharedAllocationRecord<KokkosResilience::ResHostSpace, void> *
SharedAllocationRecord<KokkosResilience::ResHostSpace, void>::get_record(void *alloc_ptr) {
  typedef SharedAllocationHeader Header;
  typedef SharedAllocationRecord<KokkosResilience::ResHostSpace, void> RecordHost;

  SharedAllocationHeader const *const head =
      alloc_ptr ? Header::get_header(alloc_ptr) : nullptr;
  RecordHost *const record =
      head ? static_cast<RecordHost *>(head->m_record) : nullptr;

  if (!alloc_ptr || record->m_alloc_ptr != head) {
    Kokkos::Impl::throw_runtime_exception(
        std::string("Kokkos::Impl::SharedAllocationRecord< KokkosResilience::ResHostSpace , "
                    "void >::get_record ERROR"));
  }

  return record;
}

// Iterate records to print orphaned memory ...
#ifdef KOKKOS_DEBUG
void SharedAllocationRecord<KokkosResilience::ResHostSpace, void>::print_records(
    std::ostream &s, const KokkosResilience::ResHostSpace &, bool detail) {
  SharedAllocationRecord<void, void>::print_host_accessible_records(
      s, "HostSpace", &s_root_record, detail);
}
#else
void SharedAllocationRecord<KokkosResilience::ResHostSpace, void>::print_records(
    std::ostream &, const KokkosResilience::ResHostSpace &, bool) {
  throw_runtime_exception(
      "SharedAllocationRecord<ResHostSpace>::print_records only works with "
      "KOKKOS_DEBUG enabled");
}
#endif

}  // namespace Impl
}  // namespace Kokkos

/*-------------------------
ATOMIC LOCKS HERE IF NEEDED
-------------------------*/
