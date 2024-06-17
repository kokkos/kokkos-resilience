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
#include <gtest/gtest.h>
#include "TestCommon.hpp"
#include <Kokkos_Core.hpp>
#include <resilience/Resilience.hpp>

#ifdef KR_ENABLE_HDF5_PARALLEL
#include <mpi.h>
#endif

template< typename ExecSpace >
class TestViewCheckpoint : public ::testing::Test
{
public:

  using exec_space = ExecSpace;
};


namespace {

  template < typename ExecSpace, typename CpFileSpace >
  struct TestCheckPointView {
    static bool consistency_check(int iter) {
      if (iter == 50)
        return false;
      else
        return true;
    }

    static void test_view_chkpt( int iter, std::string view_prefix, int dim0, int dim1, std::string default_path ) {
      int N = 1;
      typedef Kokkos::LayoutLeft       Layout;
      typedef Kokkos::HostSpace        defaultMemSpace;  // default device
      typedef CpFileSpace              fileSystemSpace;  // file system

      fileSystemSpace::set_default_path(default_path );
      fileSystemSpace fs;
      typedef Kokkos::View<double**, Layout, defaultMemSpace> local_view_type;

      std::string viewAName = view_prefix;
      viewAName += (std::string)"_A";
      std::string viewBName = view_prefix;
      viewBName += (std::string)"_B";
#ifdef KR_ENABLE_HDF5_PARALLEL
      char buff[16];
       int mpi_rank = 0;
       MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
       sprintf( buff, ".%d", mpi_rank );
       viewAName += (std::string)buff;
       viewBName += (std::string)buff;
#endif

      local_view_type A(viewAName, dim0, dim1);
      local_view_type B(viewBName, dim0, dim1);
      local_view_type::HostMirror h_A = Kokkos::create_mirror_view(A);
      local_view_type::HostMirror h_B = Kokkos::create_mirror_view(B);

      auto F_A = Kokkos::create_chkpt_mirror(fs, h_A);
      auto F_B = Kokkos::create_chkpt_mirror(fs, h_B);

      if ( iter == 0 ) {
        Kokkos::parallel_for (Kokkos::RangePolicy<ExecSpace>(0, dim0), KOKKOS_LAMBDA(const int i) {
          for (int j=0; j< dim1; j++) {
            A(i,j) = 0;  B(i,j) = 0;
          }
        });
        Kokkos::deep_copy(h_A, A);  Kokkos::deep_copy(h_B, B);
      } else {
        fileSystemSpace::restore_all_views();  // restart from existing…
      }

      for ( int r = 0; r < N; r++ ) {
        Kokkos::deep_copy(A, h_A);  Kokkos::deep_copy(B, h_B);

        Kokkos::parallel_for (Kokkos::RangePolicy<ExecSpace>(0, dim0), KOKKOS_LAMBDA(const int i) {
          for (int j=0; j< dim1; j++) {
            A(i,j) += 1;  B(i,j) += 1;
          }
        });
        Kokkos::deep_copy(h_A, A);  Kokkos::deep_copy(h_B, B);

        if (!consistency_check(iter)) {
          fileSystemSpace::restore_view(viewAName); // restore data
          fileSystemSpace::restore_view(viewBName); // restore data
        } else {
          fileSystemSpace::checkpoint_views();  // save result
        }
      }

      for ( int i = 0; i < dim0; i++ ) {
        for (int j = 0; j < dim1; j++) {
          ASSERT_EQ( A(i,j), iter+1 );
          ASSERT_EQ( B(i,j), iter+1 );
        }
      }
    }

  };


  template < typename ExecSpace, typename CpFileSpace >
  struct TestFSConfig {

    typedef typename ExecSpace::memory_space     memory_space;

    static void test_view_chkpt(std::string file_name, int dim0, int dim1) {

      typedef Kokkos::View<char**,memory_space> Rank2ViewType;
      Rank2ViewType view_2;
      view_2 = Rank2ViewType("memory_view_2", dim0, dim1);
      typename Rank2ViewType::HostMirror h_view_2 = Kokkos::create_mirror(view_2);

      typedef CpFileSpace cp_file_space_type;
      cp_file_space_type::set_default_path("./data");
      Kokkos::View<char**,cp_file_space_type> cp_view(file_name, dim0, dim1);

      Kokkos::parallel_for (Kokkos::RangePolicy<ExecSpace>(0, dim0), KOKKOS_LAMBDA (const int i) {
        for (int j = 0; j < dim1; j++) {
          view_2(i,j) = i * dim0 + j;
        }
      });
      Kokkos::deep_copy( h_view_2, view_2 );
#ifdef KR_ENABLE_HDF5_PARALLEL
      MPI_Barrier(MPI_COMM_WORLD);
#endif

      // host_space to ExecSpace
      Kokkos::deep_copy( cp_view, h_view_2 );
      Kokkos::fence();

      Kokkos::parallel_for (Kokkos::RangePolicy<ExecSpace>(0, dim0), KOKKOS_LAMBDA (const int i) {
        for (int j = 0; j < dim1; j++) {
          view_2(i,j) = 0;
        }
      });
      Kokkos::deep_copy( h_view_2, view_2 );

      // ExecSpace to host_space
      Kokkos::deep_copy( h_view_2, cp_view );
      Kokkos::fence();

#ifdef KR_ENABLE_HDF5_PARALLEL
      MPI_Barrier(MPI_COMM_WORLD);
#endif

      for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
          ASSERT_EQ(h_view_2(i,j), i * dim0 + j);
        }
      }

    }

  };


  template < typename ExecSpace, typename CpFileSpace >
  struct TestFSDeepCopy {


    typedef typename ExecSpace::memory_space     memory_space;


    static void test_view_chkpt(std::string file_name, int dim0, int dim1) {

      typedef Kokkos::View<double**,memory_space> Rank2ViewType;
      Rank2ViewType view_2;
      view_2 = Rank2ViewType("memory_view_2", dim0, dim1);
      typename Rank2ViewType::HostMirror h_view_2 = Kokkos::create_mirror(view_2);

      typedef CpFileSpace cp_file_space_type;
      Kokkos::View<double**,cp_file_space_type> cp_view(file_name, dim0, dim1);

      Kokkos::parallel_for (Kokkos::RangePolicy<ExecSpace>(0, dim0), KOKKOS_LAMBDA (const int i) {
        for (int j = 0; j < dim1; j++) {
          view_2(i,j) = i + j;
        }
      });
      Kokkos::deep_copy( h_view_2, view_2 );

      // host_space to ExecSpace
      Kokkos::deep_copy( cp_view, h_view_2 );
      Kokkos::fence();

      Kokkos::parallel_for (Kokkos::RangePolicy<ExecSpace>(0, dim0), KOKKOS_LAMBDA (const int i) {
        for (int j = 0; j < dim1; j++) {
          view_2(i,j) = 0;
        }
      });
      Kokkos::deep_copy( h_view_2, view_2 );

      // ExecSpace to host_space
      Kokkos::deep_copy( h_view_2, cp_view );
      Kokkos::fence();

      for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
          ASSERT_EQ(h_view_2(i,j), i + j);
        }
      }


    }

  };


} // namespace


TYPED_TEST_SUITE( TestViewCheckpoint, enabled_exec_spaces );

TYPED_TEST( TestViewCheckpoint, stdio )
{
  using exec_space = typename TestFixture::exec_space;

  TestFSDeepCopy< exec_space, KokkosResilience::StdFileSpace >::test_view_chkpt(KR_TEST_DATADIR "/cp_view.bin",10,10);
  remove(KR_TEST_DATADIR "/cp_view.bin");
  TestFSDeepCopy< exec_space, KokkosResilience::StdFileSpace >::test_view_chkpt(KR_TEST_DATADIR "/cp_view.bin",100,100);
  remove(KR_TEST_DATADIR "/cp_view.bin");
  TestFSDeepCopy< exec_space, KokkosResilience::StdFileSpace >::test_view_chkpt(KR_TEST_DATADIR "/cp_view.bin",10000,10000);
  remove(KR_TEST_DATADIR "/cp_view.bin");

  mkdir(KR_TEST_DATADIR "/stdfile", 0777);
  remove(KR_TEST_DATADIR "/stdfile/view_A");
  remove(KR_TEST_DATADIR "/stdfile/view_B");
  for (int n = 0; n < 10; n++) {
    TestCheckPointView< exec_space, KokkosResilience::StdFileSpace >::test_view_chkpt(n, "view", 10,10, KR_TEST_DATADIR "/stdfile/");
  }
}


#ifdef KR_ENABLE_HDF5

TYPED_TEST( TestViewCheckpoint, hdf5 )
{
  using exec_space = typename TestFixture::exec_space;

  mkdir("./data", 0777);
  std::string file_name = KR_TEST_DATADIR "/cp_view.hdf";
#ifdef KR_ENABLE_HDF5_PARALLEL
  char buff[16];
  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  sprintf( buff, ".%d", mpi_rank );
  file_name += (std::string)buff;
#endif
  TestFSDeepCopy< exec_space, KokkosResilience::HDF5Space >::test_view_chkpt(file_name,10,10);
  remove(KR_TEST_DATADIR "/cp_view.hdf*");
  TestFSDeepCopy< exec_space, KokkosResilience::HDF5Space >::test_view_chkpt(KR_TEST_DATADIR "/cp_view.hdf",100,100);
  remove(KR_TEST_DATADIR "/cp_view.hdf*");
  TestFSDeepCopy< exec_space, KokkosResilience::HDF5Space >::test_view_chkpt(KR_TEST_DATADIR "/cp_view.hdf",10000,10000);
  remove(KR_TEST_DATADIR "/cp_view.hdf*");
  TestFSConfig< exec_space, KokkosResilience::HDF5Space >::test_view_chkpt("1D_regular_test",10,10);

  mkdir(KR_TEST_DATADIR "/hdf5", 0777);
  remove(KR_TEST_DATADIR "/hdf5/view_A*");
  remove(KR_TEST_DATADIR "/hdf5/view_B*");
  for (int n = 0; n < 10; n++) {
     TestCheckPointView< exec_space, KokkosResilience::HDF5Space >::test_view_chkpt(n, "view", 10,10,KR_TEST_DATADIR "/hdf5/");
  }
}

#endif
