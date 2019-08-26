#ifndef INC_RESILIENCE_UTIL_TRACE_HPP
#define INC_RESILIENCE_UTIL_TRACE_HPP

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

#include "Timer.hpp"

namespace KokkosResilience
{
  namespace Util
  {
    namespace detail
    {
      inline picojson::value make_json_value( int _val )
      {
        return picojson::value( static_cast< double >( _val ));
      }
  
      inline picojson::value make_json_value( double _val )
      {
        return picojson::value( _val );
      }
  
      inline picojson::value make_json_value( const std::string &_val )
      {
        return picojson::value( _val );
      }
      
      class TraceStack;
    }
    
    class TraceBase
    {
    public:
      explicit TraceBase( ) : m_done( false ), m_parent( nullptr ), m_trace_stack( nullptr )
      {}
  
      explicit TraceBase( detail::TraceStack *trace )
        : m_done( false ), m_parent( nullptr ), m_trace_stack( trace )
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
  
      virtual picojson::object get_json_object() const
      {
        picojson::object obj;
  
        obj["type"] = picojson::value( get_typestring() );
        
        auto children = picojson::array{};
        children.reserve( m_children.size() );
        
        for ( auto &&c : m_children )
        {
          children.emplace_back( picojson::value( c->get_json_object() ) );
        }
        
        obj["subtraces"] = picojson::value( children );
        
        return obj;
      }
  
      virtual void end() = 0;
      
      detail::TraceStack *trace_stack() const noexcept { return m_trace_stack; }
  
    private:
    
      bool m_done;
      
      TraceBase *m_parent;
      std::vector< std::unique_ptr< TraceBase > > m_children;
      
      detail::TraceStack *m_trace_stack;
    };
    
    namespace detail
    {
      class TraceStack
      {
      public:
        
        TraceStack()
          : m_current( nullptr )
        {}
        
        ~TraceStack() = default;
        
        TraceStack( const TraceStack & ) = delete;
        TraceStack( TraceStack && ) = default;
        
        TraceStack &operator=( const TraceStack & ) = delete;
        TraceStack &operator=( TraceStack && ) = default;
    
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

    class TraceShell : public TraceBase
    {
       public:
          inline virtual void end() {
          }

          inline virtual std::string get_typestring() const {
             return (std::string)"unknown";
          }
    };

    template< typename Id >
    class Trace : public TraceBase
    {
    public:
      
      using id_type = Id;
      
      Trace( detail::TraceStack *trace, id_type id )
        : TraceBase( trace ), m_id( std::move( id ) )
      {
        m_start_time = std::chrono::system_clock::now();
      }
      
      id_type id() const noexcept { return m_id; }
      
      picojson::object get_json_object() const override
      {
        auto obj = TraceBase::get_json_object();
        
        obj["name"] = detail::make_json_value( m_id );
        
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
        if ( m_trace )
        {
          m_trace->mark_done();
          m_trace->trace_stack()->try_pop( m_trace );
          m_trace = nullptr;
        }
      }
      
      ~TraceHandle()
      {
        end();
      }
      
    private:
      
      Trace< Id > *m_trace;
    };

#ifdef KR_ENABLE_TRACING
    template< typename TraceType, typename Context, typename... Args >
    auto begin_trace( Context &ctx, Args &&... args )
    {
      auto tr = std::make_unique< TraceType >( &ctx.trace(), std::forward< Args >( args )... );
      auto ret = TraceHandle< typename TraceType::id_type >{ tr.get() };
      ctx.trace().push( std::move( tr ) );
      
      return ret;
    }
#else
    template< typename TraceType, typename Context, typename... Args >
    auto begin_trace( Context &ctx, Args &&... args )
    {
       return TraceShell();
    }
#endif
    
    template< typename Id >
    class TimingTrace : public Trace< Id >
    {
    public:
      
      explicit TimingTrace( detail::TraceStack *trace, Id id )
        : Trace< Id >( trace, std::move( id ) ), m_timer( true ), m_duration{}
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
  
  
    template< typename Id >
    class IterTimingTrace : public TimingTrace< Id >
    {
    public:
  
      IterTimingTrace( detail::TraceStack *trace, Id id, int iteration )
        : TimingTrace< Id >( trace, std::move( id ) ), m_iteration( iteration )
      {}
  
      picojson::object get_json_object() const override
      {
        auto ret = TimingTrace< Id >::get_json_object();
    
        ret["iteration"] = detail::make_json_value( m_iteration );
    
        return ret;
      }

    private:
  
      int m_iteration;
    };
  }
}

#endif  // INC_RESILIENCE_UTIL_TRACE_HPP
