#include "Context.hpp"
#include <fstream>
#include <chrono>
#if defined(KR_ENABLE_VELOC)
   #include "veloc/VelocBackend.hpp"
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
      auto &filter = filter_opt.get();
      if ( filter["type"].as< std::string >() == "time" )
      {
        m_default_filter = Filter::TimeFilter( std::chrono::seconds{ static_cast< long >( filter["interval"].as< double >() ) } );
      }
    }
  }

  std::unique_ptr< ContextBase >
  make_context( MPI_Comm comm, const std::string &config )
  {
    auto cfg = Config{ config };

    // Check backend
    if ( cfg["backend"].as< std::string >() == "veloc" )
    {
#if defined(KR_ENABLE_VELOC)
      return std::make_unique< Context< VeloCMemoryBackend > >( comm, cfg );
#else
      return std::unique_ptr< ContextBase >{};
#endif
    } else {
      return std::unique_ptr< ContextBase >{};
    }

  }
}
