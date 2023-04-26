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

#ifndef RES_CUDA_HPP
#define RES_CUDA_HPP

#include <Kokkos_Macros.hpp>
#if defined(KR_ENABLE_CUDA)

#include <Kokkos_Core.hpp> //attempt to get rid of header errors later

#include <iosfwd>
#include <vector>

#include "ResCudaSpace.hpp" //Resilient

/*--------------------------------------------------------------------------*/

namespace KokkosResilience{

class ResCuda : public Kokkos::Cuda {
 public:

  // Type declarations for execution spaces
  using execution_space      = ResCuda;          // Tag class as Kokkos execution space

#if defined(KOKKOS_ENABLE_CUDA_UVM)
  using memory_space         = Kokkos::CudaUVMSpace;  // Preferred memory space UVM if enabled
#else
  using memory_space         = ResCudaSpace;     // Preferred memory space
#endif
  using device_type          = Kokkos::Device<execution_space, memory_space>; // Device type
  using size_type            = memory_space::size_type; // Memory size best suited
  using array_layout         = Kokkos::LayoutLeft;              // Preferred array layout
  using scratch_memory_space = Kokkos::ScratchMemorySpace<Cuda>;

  // True if and only if this method is being called in a
  // thread-parallel function.
    KOKKOS_INLINE_FUNCTION static int in_parallel() {
#if defined(__CUDA_ARCH__)
    return true;
#else
    return false;
#endif
  }

/*------------------------------------------------------------------------*/

  // Cuda space instances
  ~ResCuda() {}
  ResCuda();

  ResCuda(cudaStream_t stream, bool manage_stream = false);

  ResCuda( ResCuda && ) = default;
  ResCuda( const ResCuda & ) = default;
  ResCuda & operator = ( ResCuda && ) = default;
  ResCuda & operator = ( const ResCuda & ) = default;

/*------------------------------------------------------------------------*/
  static const char* name();

}; // template class ResCuda execution space

}  // namespace KokkosResilience

/*------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl{
/*
// templated to pass to CUdasapace
template < typename OtherSpace >
struct MemorySpaceAccess <KokkosResilience::ResCudaSpace, OtherSpace> 
     : MemorySpaceAccess <Kokkos::CudaSpace, OtherSpace>{
};

template < typename OtherSpace >
struct MemorySpaceAccess <OtherSpace, KokkosResilience::ResCudaSpace> 
     : MemorySpaceAccess <OtherSpace, Kokkos::CudaSpace>{
};

template <>
struct MemorySpaceAccess <KokkosResilience::ResCudaSpace, KokkosResilience::ResCudaSpace> 
     : MemorySpaceAccess <Kokkos::CudaSpace, Kokkos::CudaSpace>{
};
*/
} // Impl
} // Kokkos 




// Specialization accessibility information

/*------------------------------------------------------------------------*/

// This is the specialization which corrects the profiling error.

namespace Kokkos {
namespace Tools {
namespace Experimental {
template <>
struct DeviceTypeTraits<KokkosResilience::ResCuda> {
  //An ID to differentiate (for example) Serial from OpenMP in Tooling
  static constexpr DeviceType id = DeviceType::Cuda;
  static int device_id(const Cuda& exec) { return exec.cuda_device(); }
};
}  // namespace Experimental
}  // namespace Tools
}  // namespace Kokkos

/*------------------------------------------------------------------------*/

//#include "CudaResParallel.hpp" // Resilient specific parallel functors

/*------------------------------------------------------------------------*/

#endif // #if defined( KR_ENABLE_CUDA ) 
#endif // #RES_CUDA_HPP

