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
#define KOKKOS_ENABLE_MANUAL_CHECKPOINT
#include <resilience/Resilience.hpp>

#ifdef KR_ENABLE_HDF5

template< typename ExecSpace >
class TestHDF5Configuration : public ::testing::Test
{
public:
  
  using exec_space = ExecSpace;
};


namespace {
  
  struct TestHDF5ConfigParser {
    
    const char * path = "parser_test";
    hsize_t simple_var[2];  // maximum dimensions = 2
    hsize_t two_var[2];  // maximum dimensions = 2
    hsize_t var_comb[2];  // maximum dimensions = 2
    hsize_t inner_comb_two[2];  // maximum dimensions = 2
    hsize_t inner_comb_three[2];  // maximum dimensions = 2
    hsize_t inner_comb_three_adv[2];  // maximum dimensions = 2
    hsize_t inner_comb_three_adv2[2];  // maximum dimensions = 2
    KokkosResilience::KokkosHDF5ConfigurationManager config_;
    
    TestHDF5ConfigParser() :
             simple_var{0,0},
             two_var{0,0},
             var_comb{0,0},
             inner_comb_two{0,0},
             inner_comb_three{0,0},
             inner_comb_three_adv{0,0},
             inner_comb_three_adv2{0,0},
             config_( KokkosResilience::KokkosIOConfigurationManager::
                           get_instance()->get_config(path) ) {
    }


    void test_config() {
      
       boost::property_tree::ptree l_config = config_.get_config()->get_child("Layout_Config");
       std::map<const std::string, size_t> var_list;
       var_list["SIMPLE_VAR"] = 5;
       var_list["VAR_A"] = (size_t)1;
       var_list["VAR_B"] = (size_t)2;
       var_list["VAR_C"] = (size_t)3;

       // {SIMPLE_VAR} 
       config_.set_param_list( l_config, 0, "simple_var", simple_var, var_list );
       
       // {VAR_A} * {VAR_B}
       config_.set_param_list( l_config, 0, "two_var", two_var, var_list );
       
       // {VAR_B}*({VAR_A}+2)
       config_.set_param_list( l_config, 0, "var_comb", var_comb, var_list );
       
       // (({VAR_A}*2)*{VAR_B})+2
       config_.set_param_list( l_config, 0, "inner_comb_two", inner_comb_two, var_list );

       // ((({VAR_A}*2)*{VAR_B})+2)*{VAR_C}
       config_.set_param_list( l_config, 0, "inner_comb_three", inner_comb_three, var_list );

       // ((({VAR_A}*2)*{VAR_B})+2)*({VAR_C}+2)
       config_.set_param_list( l_config, 0, "inner_comb_three_adv", inner_comb_three_adv, var_list );
      
       // (((({VAR_A}*2)*{VAR_B})+2)+({VAR_C}+2))*2
       config_.set_param_list( l_config, 0, "inner_comb_three_adv2", inner_comb_three_adv2, var_list );
       
       ASSERT_EQ(simple_var[0], 5);
       ASSERT_EQ(two_var[0], 3);
       ASSERT_EQ(var_comb[0], 6);
       ASSERT_EQ(inner_comb_two[0], 6);
       ASSERT_EQ(inner_comb_three[0], 18);
       ASSERT_EQ(inner_comb_three_adv[0], 30);
       ASSERT_EQ(inner_comb_three_adv2[0], 22);
    }
    
  };
  
  
} // namespace


TYPED_TEST_SUITE( TestHDF5Configuration, enabled_exec_spaces );

TYPED_TEST( TestHDF5Configuration, hdf5_configuration )
{
  
  TestHDF5ConfigParser hcp;
  hcp.test_config();

}

#endif
