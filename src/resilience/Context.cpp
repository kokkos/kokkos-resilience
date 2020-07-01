#include "Context.hpp"
#include <fstream>
#include <chrono>
#include <stdexcept>

namespace KokkosResilience
{
  ContextBase::ContextBase( Config cfg )
      : m_config( std::move( cfg ) ),
        m_default_filter{ Filter::DefaultFilter{} }
  {
    auto filter_opt = m_config.get( "filter" );

    if ( filter_opt )
    {
      auto &filter = filter_opt.get();
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

  std::unique_ptr< ContextBase >
  make_context( const std::string &config )
  {
    auto cfg = Config{ config };
    return std::unique_ptr< ContextBase >{};
  }
}