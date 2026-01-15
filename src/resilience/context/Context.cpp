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
#include "../backend/StdFileBackend.hpp"
#ifdef KR_ENABLE_VELOC_BACKEND  
#include "resilience/backend/VelocBackend.hpp"
#endif

namespace KokkosResilience
{
  ContextBase::ContextBase( Config cfg )
      : m_config( std::move( cfg ) ),
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

  char* ContextBase::get_scratch_buffer(size_t minimum_size)
  {
    auto view = m_scratch_buffer.view();
    // view size is the number of characters ever initialized in the stream's
    // internal buffer, independent of current tellp/tellg values
    if(view.size() < minimum_size)
    {
      // Write any character to force buffer space to grow, or force more
      // characters to be initialized in existing buffer
      auto pos = m_scratch_buffer.tellp();
      m_scratch_buffer.seekp(minimum_size-1);
      m_scratch_buffer.put('\0');
      view = m_scratch_buffer.view();
      m_scratch_buffer.seekp(pos);
    }
    assert(view.size() >= minimum_size);

    return const_cast<char*>(view.data());
  }

  std::stringstream& ContextBase::get_scratch_stream()
  {
    m_scratch_buffer.seekp(0);
    m_scratch_buffer.seekg(0);
    return m_scratch_buffer;
  }

  std::unique_ptr< ContextBase >
  make_context( const std::string &config )
  {
    return make_context( Config{ config } );
  }

  std::unique_ptr< ContextBase >
  make_context( const Config &cfg )
  {
    using fun_type = std::function<std::unique_ptr<ContextBase>()>;
    static std::unordered_map<std::string, fun_type> backends = {
        {"stdfile", [&]() -> std::unique_ptr<ContextBase> {
           return std::make_unique<StdFileContext<StdFileBackend> >(cfg);
         }}};
  
    auto pos = backends.find(cfg["backend"].as<std::string>());
    if (pos == backends.end()) return {};
  
    return pos->second();
  }

#ifdef KR_ENABLE_MPI_CONTEXT
  std::unique_ptr< ContextBase >
  make_context( MPI_Comm comm, const std::string &config )
  {
    return make_context( comm, Config{ config } );
  }

  std::unique_ptr< ContextBase >
  make_context( MPI_Comm comm, const Config &cfg )
  {
    using fun_type = std::function<std::unique_ptr<ContextBase>()>;
    static std::unordered_map<std::string, fun_type> backends = {
#ifdef KR_ENABLE_VELOC_BACKEND
        {"veloc", [&]() -> std::unique_ptr<ContextBase> {
          return std::make_unique<MPIContext<VeloCMemoryBackend> >(comm, cfg);
        }},
        {"veloc-noop", [&]() -> std::unique_ptr<ContextBase> {
          return std::make_unique<MPIContext<VeloCRegisterOnlyBackend> >(comm, cfg);
        }}
#endif
      };

    auto pos = backends.find(cfg["backend"].as<std::string>());
    if (pos == backends.end()) return {};

    return pos->second();
  }
#endif
}
