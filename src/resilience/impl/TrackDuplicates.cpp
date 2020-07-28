#include <Kokkos_Core.hpp>
#include <impl/TrackDuplicates.hpp>

namespace KokkosResilience {

std::map<std::string, void*> DuplicateTracker::kernel_func_list;

void DuplicateTracker::add_kernel_func(std::string name, void* func_ptr) {
  kernel_func_list[name] = func_ptr;
}

void* DuplicateTracker::get_kernel_func(std::string name) {
  return kernel_func_list[name];
}

}  // namespace KokkosResilience

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
