#include "Config.hpp"
#include <pico/picojson.h>

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

  Config::Config( const boost::filesystem::path &p )
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