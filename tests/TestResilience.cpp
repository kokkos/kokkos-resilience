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

#ifdef KR_ENABLE_CUDA_EXEC_SPACE
#include <gtest/gtest.h>
#include "TestCommon.hpp"
#include <Kokkos_Core.hpp>
#include <resilience/Resilience.hpp>
#include <resilience/cuda/ResCudaSpace.hpp>
#include <resilience/cuda/ResCuda.hpp>

template< typename ExecSpace >
class TestResilience : public ::testing::Test
{
public:

  using exec_space = ExecSpace;
};


struct ResSurrogate {
  typedef Kokkos::View< int*, Kokkos::CudaSpace > ViewType;
//   typedef Kokkos::View< int*, Kokkos::HostSpace > ViewType;
  ViewType vt;

  KOKKOS_INLINE_FUNCTION
  ResSurrogate(const ViewType & vt_) : vt(vt_) {
    printf("inside functor constructor\n");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    vt(i) = i;
  }

};

struct testCopy {

  testCopy() {
    printf("test copy original\n");
  }

  testCopy(const testCopy &) {
    printf("test copy const\n");
  }

  void callme() const {
  }

};



template< class ExecSpace, class ScheduleType, class DataType >
struct TestResilientRange {
  typedef int value_type; ///< typedef required for the parallel_reduce

  typedef Kokkos::View< DataType*, KokkosResilience::ResCudaSpace > view_type;
//  typedef Kokkos::View< int*, Kokkos::HostSpace > test_type;
//  typedef Kokkos::View< int*, Kokkos::HostSpace > view_type;

  int N;

  TestResilientRange( const size_t N_ )
    :  N(N_)
  {}

  void test_for()
  {
//     printf("dup kernel ptr start = %08x \n", Kokkos::Experimental::CombineFunctor<int, Kokkos::ResCuda>::s_dup_kernel);

//     view_type m_data ( Kokkos::ViewAllocateWithoutInitializing( "data" ), N );
    view_type m_data ( "data", N );
//     test_type t_data ( "test", N );
    typename view_type::HostMirror v = Kokkos::create_mirror_view(m_data);
//      for (int i = 0; i < N; i++) {
//         v(i) = i;
//      }
//      Kokkos::deep_copy( m_data, v );

//      ResSurrogate f(m_data);
    printf("calling parallel_for\n");
//      Kokkos::parallel_for(N, f);
    Kokkos::RangePolicy<KokkosResilience::ResCuda> rp (0,N);
/*      auto ml = KOKKOS_LAMBDA(const int i){
   #if defined(__CUDA_ARCH__)
         printf("insided lambda[%d]\n", i);
   #endif
         m_data(i)=i;
      };
*/
    Kokkos::parallel_for(rp,KOKKOS_LAMBDA(const int i){
      m_data(i)=i;
    });
    Kokkos::fence();

    Kokkos::deep_copy(v, m_data);

    for (int i = 0; i < N; i++) {
      ASSERT_EQ(v(i), i );
    }
  }

};

KR_DECLARE_RESILIENCE_OBJECTS(int,int)


TYPED_TEST_SUITE( TestResilience, enabled_exec_spaces );

TYPED_TEST( TestResilience, range )
{

  using exec_space = typename TestFixture::exec_space;

  KR_ADD_RESILIENCE_OBJECTS(int,int);

  { TestResilientRange< exec_space, Kokkos::Schedule<Kokkos::Static>, int >f(10); f.test_for(); }
}

#endif
