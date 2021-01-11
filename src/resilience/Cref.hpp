#ifndef INC_RESILIENCE_CREF_HPP
#define INC_RESILIENCE_CREF_HPP

#include <vector>

namespace KokkosResilience
{
  namespace Detail
  {
    struct CrefImpl
    {
      CrefImpl( void *p, std::size_t s, std::size_t n, const char *_name )
          : ptr( p ), sz( s ), num( n ), name( _name )
      {}

      void *ptr;
      std::size_t sz;
      std::size_t num;
      const char *name;
    };

    struct Cref : public CrefImpl
    {
      using CrefImpl::CrefImpl;

      Cref( const Cref &_other )
          : CrefImpl( _other.ptr, _other.sz, _other.num, _other.name )
      {
        if ( check_ref_list )
          check_ref_list->emplace_back( ptr, sz, num, name );
      }

      static std::vector< CrefImpl > *check_ref_list;
    };
  }

  template< typename T >
  auto check_ref( T &_t, const char *_str )
  {
    return Detail::Cref{ reinterpret_cast< void * >( &_t ), sizeof( T ), 1, _str };
  }
}

#endif  // INC_RESILIENCE_CREF_HPP
