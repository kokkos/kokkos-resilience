#ifndef INC_RESILIENCE_CONFIG_HPP
#define INC_RESILIENCE_CONFIG_HPP

#include <boost/variant.hpp>
#include <boost/optional.hpp>
#include <boost/filesystem.hpp>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <type_traits>

namespace KokkosResilience
{
  struct ConfigKeyError : std::runtime_error
  {
    explicit ConfigKeyError( const std::string &key )
      : std::runtime_error( "key error: " + key )
    {}
  };

  struct ConfigValueError : std::runtime_error
  {
    ConfigValueError()
        : std::runtime_error( "value error" )
    {}
  };

  class Config
  {
  public:

    class Value
    {
    public:

      using variant_type = boost::variant< double, std::string, bool >;

      Value() = default;

      template< typename T, typename = std::enable_if_t< std::is_constructible< variant_type, T >::value > >
      explicit Value( T &&val )
      {
        m_variant = std::forward< T >( val );
      }

      template< typename T >
      const T &as() const
      {
        auto *val = boost::get< T >( &m_variant );
        if ( !val )
          throw ConfigValueError();

        return *val;
      }

      template< typename T >
      void set( T &&_value )
      {
        m_variant = std::forward< T >( _value );
      }

    private:

      variant_type m_variant;
    };

    class Entry
    {
    public:

      Entry() = default;
      explicit Entry( const Value &_val )
        : m_variant( _val )
      {}

      template< typename... Args >
      void emplace( const std::string &key, Args &&... args )
      {
        if ( m_variant.which() != 0 )
          throw ConfigKeyError( key );

        auto &map = boost::get< map_type >( m_variant );
        map.emplace( std::piecewise_construct,
            std::forward_as_tuple( key ),
            std::forward_as_tuple( std::forward< Args >( args )... ) );
      }

      const Entry &operator[]( const std::string &key ) const
      {
        if ( m_variant.which() != 0 )
          throw ConfigKeyError( key );

        const auto &map = boost::get< map_type >( m_variant );

        auto pos = map.find( key );
        if ( pos == map.end() )
          throw ConfigKeyError( key );

        return pos->second;
      }

      Entry &operator[]( const std::string &key )
      {
        if ( m_variant.which() != 0 )
          m_variant = map_type{};

        auto &map = boost::get< map_type >( m_variant );

        return map[key];
      }

      boost::optional< Entry > get( const std::string &key ) const
      {
        if ( m_variant.which() != 0 )
          throw ConfigKeyError( key );

        const auto &map = boost::get< map_type >( m_variant );

        auto pos = map.find( key );
        if ( pos == map.end() )
          return boost::none;

        return pos->second;
      }

      template< typename T >
      void set( T &&val )
      {
        m_variant = Value( std::forward< T >( val ) );
      }

      template< typename T >
      const T &as() const
      {
        if ( m_variant.which() != 2 )
          throw ConfigValueError();

        const auto &val = boost::get< Value >( m_variant );
        return val.as< T >();
      }

      bool is_value() const noexcept
      {
        return m_variant.which() == 2;
      }

      bool is_object() const noexcept
      {
        return m_variant.which() == 0;
      }

    private:

      using map_type = std::unordered_map< std::string, Entry >;
      using array_type = std::vector< Entry >;

      boost::variant< boost::recursive_wrapper< map_type >,
                      boost::recursive_wrapper< array_type >, Value > m_variant;
    };

    Config() = default;
    explicit Config( const boost::filesystem::path &p );

    const Entry &operator[]( const std::string &key ) const
    {
      return m_root[key];
    }

    Entry &operator[]( const std::string &key )
    {
      return m_root[key];
    }

    auto
    get( const std::string &key ) const
    {
      return m_root.get( key );
    }
    template< typename T >
    void set( const std::string &key, T &&val )
    {
      m_root.set( key, std::forward< T >( val ) );
    }

  private:

    Entry m_root;
  };
}

#endif  // INC_RESILIENCE_CONFIG_HPP
