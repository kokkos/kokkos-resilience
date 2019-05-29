#include "trace.hpp"

namespace KokkosResilience
{
  namespace Util
  {
    namespace detail
    {
      thread_local TraceStack TraceStack::instance;
    }
  }
}