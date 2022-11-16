#include "Registration.hpp"
#include "Context.hpp"

namespace KokkosResilience
{
  namespace Detail
  {
  }
}

namespace std { 
  bool operator==(const KokkosResilience::Registration& lhs, const KokkosResilience::Registration& rhs){
    return (*lhs.get()) == (*rhs.get());
  }
}
