#ifndef INC_KOKKOS_RESILIENCE_RESILIENTREF_HPP
#define INC_KOKKOS_RESILIENCE_RESILIENTREF_HPP

#include <utility>
#include <vector>
#include <memory>
#include "Utility.hpp"

namespace KokkosResilience
{
  template< typename T >
  class Ref
  {
   public:

    using reference = T &;
    using pointer = T *;

    explicit Ref( reference _obj )
      : m_object( &_obj ) {

    }

    Ref( Ref &&_other ) noexcept = default;
    Ref &operator=( Ref &&_other ) noexcept = default;

    Ref( const Ref &_other )
      : m_object( _other.m_object ) {
      // Now explicitly copy the object to a temporary
      auto copy = *m_object;
    }

    Ref &operator=( const Ref &_other ) {
      m_object = _other.m_object;
      auto copy = *m_object;
    }

    reference get() const noexcept { return *m_object; }
    reference operator*() const noexcept { return *m_object; }
    pointer operator->() const noexcept { return m_object; }

   private:

    T *m_object;
  };

  template< typename T, typename... Args >
  Ref< T > make_ref( Args &&... _args )
  {
    return Ref< T >( in_place_t{}, std::forward< Args >( _args )... );
  }
}

#endif  // KOKKOS_RESILIENCE_RESILIENTREF_HPP
