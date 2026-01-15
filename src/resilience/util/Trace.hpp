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

namespace KokkosResilience::Util::inline Trace
{
#ifndef KR_ENABLE_TRACING
  // These classes are never constructed, just passed as TraceType parameters
  template< typename Id >
  class TimingTrace;
  template< typename Id >
  class IterTimingTrace;

  template<
    template<typename> class TraceType = TimingTrace, typename Context,
    typename Id, typename... Args
  >
  constexpr auto begin_trace(Context &ctx, Id &&id, Args &&... args);

  template< typename Id >
  class TraceHandle
  {
  public:
    constexpr void end(){}
    constexpr ~TraceHandle(){}
  protected:
    template<
      template<typename> class TraceType, typename Context,
      typename IdType, typename... Args
    >
    friend constexpr auto begin_trace(Context&, IdType&&, Args&&...);

    constexpr explicit TraceHandle(){}
  };

  template<
    template<typename> class TraceType, typename Context,
    typename Id, typename... Args
  >
  constexpr auto begin_trace(Context &ctx, Id &&id, Args &&... args)
  {
    return TraceHandle< Id >();
  }

  class TraceStack
  {
  public:
    constexpr TraceStack(){}

    TraceStack( const TraceStack & ) = delete;
    constexpr TraceStack( TraceStack && ){}
    TraceStack &operator=( const TraceStack & ) = delete;
    constexpr TraceStack &operator=( TraceStack && ){ return *this; }

    constexpr std::ostream &write( std::ostream &strm ){ return strm; }
    constexpr void write( int ){ }
  };
#else // KR_ENABLE_TRACING
  template< typename Id >
  class TraceHandle;

  template< typename Id >
  class TimingTrace;

  template<
    template<typename> class TraceType = TimingTrace, typename Context,
    typename Id, typename... Args
  >
  auto begin_trace(Context &ctx, Id &&id, Args &&... args);

  class TraceStack;

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

    class TraceBase
    {
    public:
      explicit TraceBase( )
        : m_done( false ), m_parent( nullptr ), m_trace_stack( nullptr )
      {}

      explicit TraceBase( TraceStack *trace )
        : m_done( false ), m_parent( nullptr ), m_trace_stack( trace )
      {}

      virtual ~TraceBase() = default;

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

      TraceStack *trace_stack() const noexcept { return m_trace_stack; }

    private:

      bool m_done;

      TraceBase *m_parent;
      std::vector< std::unique_ptr< TraceBase > > m_children;

      TraceStack *m_trace_stack;
    };
  } // namespace detail

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

    void write( int pid )
    {
      std::string fname = "trace" + std::to_string(pid) + ".json";
      std::cout << "writing trace to " << fname << '\n';

      std::ofstream out(fname);
      write(out);
    }
  protected:
    template<
      template<typename> class TraceType, typename Context,
      typename Id, typename... Args
    >
    friend auto begin_trace(Context&, Id&&, Args&&...);

    void push( std::unique_ptr< detail::TraceBase > &&tr )
    {
      if ( m_current )
      {
        m_current = m_current->add_child( std::move( tr ) );
      } else {
        m_traces.emplace_back( std::move( tr ) );
        m_current = m_traces.back().get();
      }
    }

    template< typename Id >
    friend class TraceHandle;

    void try_pop( detail::TraceBase *tr )
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

  private:

    std::vector< std::unique_ptr< detail::TraceBase > > m_traces;

    detail::TraceBase *m_current;
  };


  namespace detail
  {
    template< typename Id >
    class Trace : public TraceBase
    {
    public:

      using id_type = Id;

      Trace( TraceStack *trace, id_type id )
        : TraceBase( trace ), m_id( std::move( id ) )
      {
        m_start_time = std::chrono::system_clock::now();
      }

      virtual ~Trace() = default;

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
  } // namespace detail

  template< typename Id >
  class TraceHandle
  {
  public:
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

  protected:
    template<
      template<typename> class TraceType, typename Context,
      typename IdType, typename... Args
    >
    friend auto begin_trace(Context&, IdType&&, Args&&...);

    explicit TraceHandle( detail::Trace< Id > *tr )
      : m_trace( tr )
    {}

  private:

    detail::Trace< Id > *m_trace;
  };

  template<
    template<typename> class TraceType, typename Context,
    typename Id, typename... Args
  >
  auto begin_trace(Context &ctx, Id &&id, Args &&... args)
  {
    auto tr = std::make_unique< TraceType< Id > >(
      &ctx.trace(), std::forward<Id>(id), std::forward< Args >( args )...
    );
    auto ret = TraceHandle< Id >{ tr.get() };
    ctx.trace().push( std::move( tr ) );

    return ret;
  }

  template< typename Id >
  class TimingTrace : public detail::Trace< Id >
  {
  public:

    explicit TimingTrace( TraceStack *trace, Id id )
      : detail::Trace< Id >( trace, std::move( id ) )
    {
    }

    void end() override
    {
      m_duration = m_timer.time();

      detail::Trace< Id >::end();
    }

    picojson::object get_json_object() const override
    {
      auto ret = detail::Trace< Id >::get_json_object();

      auto time_seconds = std::chrono::duration< double >( m_duration );
      ret["time"] = detail::make_json_value( time_seconds.count() );

      return ret;
    }

    std::string get_typestring() const override
    {
      return "timing";
    }

  private:

    Timer m_timer { true };
    Timer::duration_type m_duration {};
  };


  template< typename Id >
  class IterTimingTrace : public TimingTrace< Id >
  {
  public:

    IterTimingTrace( TraceStack *trace, Id id, int iteration )
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
#endif  // KR_ENABLE_TRACING
} // namespace KokkosResilience::Util::inline Trace

#endif  // INC_RESILIENCE_UTIL_TRACE_HPP
