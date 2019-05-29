#ifndef INC_RESILIENCE_TRACE_HPP
#define INC_RESILIENCE_TRACE_HPP

#include <string>
#include <unordered_map>
#include <memory>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <iterator>
#include <fstream>

#include <pico/picojson.h>

#include "timer.hpp"

namespace KokkosResilience
{
  namespace Util
  {
    class TraceBase
    {
    public:
  
      TraceBase()
        : m_done( false ), m_parent( nullptr )
      {}
    
      void mark_done()
      {
        m_done = true;
      }
    
      bool done()
      {
        return m_done;
      }
      
      void set_parent( TraceBase *tr )
      {
        m_parent = tr;
      }
      
      TraceBase *parent()
      {
        return m_parent;
      }
      
      TraceBase *add_child( std::unique_ptr< TraceBase > &&tr )
      {
        m_children.emplace_back( std::move( tr ) );
        auto ret = m_children.back().get();
        ret->set_parent( this );
        
        return ret;
      }
      
      virtual std::string get_typestring() const = 0;
  
      virtual picojson::object get_json_object() const = 0;
  
      virtual void end() = 0;
  
    private:
    
      bool m_done;
      
      TraceBase *m_parent;
      std::vector< std::unique_ptr< TraceBase > > m_children;
    };
    
    namespace detail
    {
      inline picojson::value make_json_value( int _val )
      {
        return picojson::value( static_cast< double >( _val ) );
      }
  
      inline picojson::value make_json_value( double _val )
      {
        return picojson::value( _val );
      }
  
      inline picojson::value make_json_value( const std::string &_val )
      {
        return picojson::value( _val );
      }
      
      class TraceStack
      {
      public:
        
        static thread_local TraceStack instance;
        
        TraceStack()
          : m_current( nullptr )
        {}
        
        ~TraceStack()
        {
          std::ostringstream fname;
          fname << "trace" << ".json";
          
          std::ofstream out( fname.str() );
          
          write( out );
        }
        
        TraceStack( const TraceStack & ) = delete;
        
        
        TraceStack &operator=( const TraceStack & ) = delete;
    
        void push( std::unique_ptr< TraceBase > &&tr )
        {
          if ( m_current )
          {
            m_current = m_current->add_child( std::move( tr ) );
          } else {
            m_traces.emplace_back( std::move( tr ) );
            m_current = m_traces.back().get();
          }
        }
    
        void try_pop( TraceBase *tr )
        {
          if ( !tr )
            return;
          
          if ( tr == m_current )
          {
            m_current->end();
            m_current = tr->parent();
          }
          
          while ( m_current && m_current->done() )
          {
            m_current->end();
            m_current = tr->parent();
          }
        }
        
        std::ostream &write( std::ostream &strm )
        {
          picojson::object root;
          
          picojson::array traces;
          traces.reserve( m_traces.size() );
          
          for ( auto &&trace : m_traces )
          {
            traces.emplace_back( picojson::value( trace->get_json_object() ) );
          }
          
          root["traces"] = picojson::value( traces );
          
          auto val = picojson::value( root );
          
          val.serialize( std::ostream_iterator< char >( strm ), true );
          
          return strm;
        }
  
      private:
        
        std::vector< std::unique_ptr< TraceBase > > m_traces;
        
        TraceBase *m_current;
      };
      
      
    }
    
    template< typename Id >
    class Trace : public TraceBase
    {
    public:
      
      using id_type = Id;
      
      Trace( id_type id )
        : m_id( std::move( id ) )
      {
        m_start_time = std::chrono::system_clock::now();
      }
      
      id_type id() const noexcept { return m_id; }
      
      picojson::object get_json_object() const override
      {
        picojson::object obj;
        
        obj["name"] = detail::make_json_value( m_id );
        obj["type"] = picojson::value( get_typestring() );
        
        auto starttm = std::chrono::system_clock::to_time_t( m_start_time );
        std::ostringstream start_str;
        start_str << std::put_time( std::localtime( &starttm ), "%T" );
        obj["start_timestamp"] = detail::make_json_value( start_str.str() );
  
        auto endtm = std::chrono::system_clock::to_time_t( m_end_time );
        std::ostringstream end_str;
        end_str << std::put_time( std::localtime( &endtm ), "%T" );
        obj["end_timestamp"] = detail::make_json_value( end_str.str() );
        
        return obj;
      }
      
      void end() override
      {
        m_end_time = std::chrono::system_clock::now();
      }
      
    private:
      
      id_type m_id;
      std::chrono::system_clock::time_point m_start_time;
      std::chrono::system_clock::time_point m_end_time;
    };
    
    template< typename Id >
    class TraceHandle
    {
    public:
      
      explicit TraceHandle( Trace< Id > *tr )
        : m_trace( tr )
      {}
      
      void end()
      {
        m_trace->mark_done();
        detail::TraceStack::instance.try_pop( m_trace );
        m_trace = nullptr;
      }
      
      ~TraceHandle()
      {
        end();
      }
      
    private:
      
      Trace< Id > *m_trace;
    };
    
    template< typename TraceType, typename... Args >
    auto begin_trace( Args &&... args )
    {
      auto tr = std::make_unique< TraceType >( std::forward< Args >( args )... );
      auto ret = TraceHandle< typename TraceType::id_type >{ tr.get() };
      detail::TraceStack::instance.push( std::move( tr ) );
      
      return ret;
    }
    
    template< typename Id >
    class TimingTrace : public Trace< Id >
    {
    public:
      
      explicit TimingTrace( Id id )
        : Trace< Id >( std::move( id ) ), m_timer( true ), m_duration{}
      {
      }
      
      void end() override
      {
        m_duration = m_timer.time();
  
        Trace< Id >::end();
      }
  
      picojson::object get_json_object() const override
      {
        auto ret = Trace< Id >::get_json_object();
  
        auto time_seconds = std::chrono::duration< double >( m_duration );
        ret["time"] = detail::make_json_value( time_seconds.count() );
        
        return ret;
      }
  
      std::string get_typestring() const override
      {
        return "timing";
      }
      
    private:
      
      Timer m_timer;
      Timer::duration_type m_duration;
    };
  }
}

#endif
