#include <Kokkos_View.hpp>
#include "DynamicViewHooks.hpp"

namespace KokkosResilience {
DynamicViewHooks::callback_overload_set DynamicViewHooks::copy_constructor_set;
DynamicViewHooks::callback_overload_set DynamicViewHooks::copy_assignment_set;
DynamicViewHooks::callback_overload_set DynamicViewHooks::move_constructor_set;
DynamicViewHooks::callback_overload_set DynamicViewHooks::move_assignment_set;
thread_local bool DynamicViewHooks::reentrant = false;
}  // namespace KokkosResilience
