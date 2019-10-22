#include "AutomaticCheckpoint.hpp"

namespace KokkosResilience
{
  namespace detail
  {
    std::vector< cref_impl > *cref::check_ref_list = nullptr;
  }
}