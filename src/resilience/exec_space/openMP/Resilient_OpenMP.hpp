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

#ifndef INC_RESILIENCE_OPENMP_RESOPENMP_HPP
#define INC_RESILIENCE_OPENMP_RESOPENMP_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_OPENMP )

#include <cstddef>
#include <iosfwd>
#include <vector>

#include <Kokkos_Core.hpp>
#include "Resilient_HostSpace.hpp" //Resilient

/*------------------------------------------------------------------------*/

namespace KokkosResilience {

// Resilient Kokkos Execution Space that uses OpenMP to run multiple
// duplicates of a desired kernel, each on a separate partition.
class ResOpenMP : public Kokkos::OpenMP {
  public:

    //Type declarations for execution spaces following API
    using execution_space      = ResOpenMP;                          //Tag class as Kokkos execution space
    using memory_space         = ResHostSpace;                       //Preferred memory space
    using device_type          = Kokkos::Device<execution_space,memory_space>; //Preferred device_type
    using size_type            = memory_space::size_type;            //Preferred size_type
    using array_layout         = Kokkos::LayoutRight;                //Preferred array layout
    using scratch_memory_space = Kokkos::ScratchMemorySpace<OpenMP>; //Preferred scratch memory

/*------------------------------------*/

    ~ResOpenMP() {}
    ResOpenMP();

    ResOpenMP( ResOpenMP && ) = default;
    ResOpenMP( const ResOpenMP & ) = default;
    ResOpenMP & operator = ( ResOpenMP && ) = default;
    ResOpenMP & operator = ( const ResOpenMP & ) = default;

/*------------------------------------*/

    static const char* name();

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
  < KokkosResilience::ResHostSpace, Kokkos::OpenMP::scratch_memory_space> : MemorySpaceAccess
  < Kokkos::HostSpace, Kokkos::OpenMP::scratch_memory_space >
{};

}  // namespace Impl
}  // namespace Kokkos

/*------------------------------------------------------------------------*/

// This is the specialization which corrects the profiling error.

namespace Kokkos {
namespace Tools {
namespace Experimental {
template <>
struct DeviceTypeTraits<KokkosResilience::ResOpenMP> {
  static constexpr DeviceType id = DeviceType::OpenMP;
  static int device_id(const OpenMP&) {return 0;}
};
}  // namespace Experimental
}  // namespace Tools
}  // namespace Kokkos

/*------------------------------------------------------------------------*/

#include "Resilient_OpenMP_Parallel_For.hpp" // Resilient specific parallel functors

/*------------------------------------------------------------------------*/

#endif //#if defined( KOKKOS_ENABLE_OPENMP )
#endif //INC_RESILIENCE_OPENMP_RESOPENMP_HPP
