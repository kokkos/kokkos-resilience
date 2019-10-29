#ifndef INC_RESILIENCE_CREF_HPP
#define INC_RESILIENCE_CREF_HPP

#include <vector>

namespace KokkosResilience
{
  namespace Detail
  {
    struct CrefImpl
    {
      CrefImpl( void *p, std::size_t s, std::size_t n )
          : ptr( p ), sz( s ), num( n )
      {}

      void *ptr;
      std::size_t sz;
      std::size_t num;
    };

    struct Cref : public CrefImpl
    {
      using CrefImpl::CrefImpl;

      Cref( const Cref &_other )
          : CrefImpl( _other.ptr, _other.sz, _other.num )
      {
        if ( check_ref_list )
          check_ref_list->emplace_back( ptr, sz, num );
      }

      static std::vector< CrefImpl > *check_ref_list;
    };
  }

  template< typename T >
  auto check_ref( T &_t )
  {
    return Detail::Cref{ reinterpret_cast< void * >( &_t ), sizeof( T ), 1 };
  }
}

#endif  // INC_RESILIENCE_CREF_HPP
