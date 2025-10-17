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
#include "Context.hpp"
#include <fstream>
#include <chrono>
#include <stdexcept>
#include "StdFileContext.hpp"
#ifdef KR_ENABLE_MPI_CONTEXT
#include "MPIContext.hpp"
#endif

#include "resilience/backend/Backend.hpp"

namespace KokkosResilience
{
  ContextBase::ContextBase( Config cfg, int proc_id )
      : m_pid( proc_id ),
        m_config( std::move( cfg ) ),
        m_default_filter{ Filter::DefaultFilter{} }
  {
    auto filter_opt = m_config.get( "filter" );

    if ( filter_opt )
    {
      auto &filter = *filter_opt;
      if ( filter["type"].as< std::string >() == "time" )
      {
        m_default_filter = Filter::TimeFilter( std::chrono::seconds{ static_cast< long >( filter["interval"].as< double >() ) } );
      } else if ( filter["type"].as< std::string >() == "iteration" ) {
        m_default_filter = Filter::NthIterationFilter( static_cast< int >( filter["interval"].as< double >() ) );
      } else if ( filter["type"].as< std::string >() == "default") {
        m_default_filter = Filter::DefaultFilter{};
      } else {
        throw std::runtime_error( "invalid filter specified" );
      }
    }
  }

  char* ContextBase::get_scratch_buffer(size_t minimum_size){
    if(m_scratch_buffer.size() < minimum_size){
      m_scratch_buffer.resize(minimum_size);
    }
    return m_scratch_buffer.data();
  }

  ContextBase::Region ContextBase::get_region(const std::string& label){
    return *regions.try_emplace(label).first;
  }

  void ContextBase::register_member(Region region, Registration& member){
    int& use_count = use_counts[member.hash()];
    if(region.members.insert(member).second) use_count++;
    if(use_count == 1) m_backend->register_member(member);
  }

  void ContextBase::deregister_member(Region region, Registration& member){
    int& use_count = use_counts[member.hash()];
    if(region.members.erase(member)){
      use_count--;
      assert(use_count >= 0);
    }
    if(use_count == 0){
      m_backend->deregister_member(member);
      use_counts.erase(member.hash());
    }
  }

  void ContextBase::reset() {
    this->reset_impl();
    regions = {};
    use_counts = {};
    active_region.reset();
    active_filter.reset();
  }

  std::unique_ptr< ContextBase >
  make_context( const std::string &config, int pid ){
    return make_context(Config{ config }, pid);
  }
  
  std::unique_ptr< ContextBase >
  make_context( Config config, int pid ){
    return std::make_unique<StdFileContext>(config, pid);
  }

#ifdef KR_ENABLE_MPI_CONTEXT
  std::unique_ptr< ContextBase >
  make_context( MPI_Comm comm, const std::string &config ){
    return make_context(comm, Config{ config });
  }
  std::unique_ptr< ContextBase >
  make_context( MPI_Comm comm, Config config ){
    return std::make_unique<MPIContext>(comm, config);
  }
#endif
}
