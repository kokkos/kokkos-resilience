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
#include "Config.hpp"
#include <pico/picojson.h>
#include <fstream>

namespace KokkosResilience
{
  namespace
  {
    Config::Entry parse_json( const picojson::object &json )
    {
      Config::Entry e;
      for ( auto &&entry : json )
      {
        if ( entry.second.is< picojson::object >() )
        {
          e.emplace( entry.first, parse_json( entry.second.get< picojson::object >() ) );
        } else if ( entry.second.is< std::string >() )
        {
          e.emplace( entry.first, Config::Value( entry.second.get< std::string >() ) );
        } else if ( entry.second.is< double >() ) {
          e.emplace( entry.first, Config::Value( entry.second.get< double >() ) );
        }
      }

      return e;
    }
  }

  Config::Config( const std::filesystem::path &p )
  {
    std::ifstream instrm{ p.string() };

    using iter_type = std::istream_iterator< char >;

    iter_type input( instrm );
    picojson::value v;
    std::string err;

    input = picojson::parse( v, input, iter_type{}, &err );
    if ( !err.empty() )
    {
      std::cerr << err << std::endl;
    }

    auto baseobj = v.get< picojson::object >();
    m_root = parse_json( baseobj );
  }
}
