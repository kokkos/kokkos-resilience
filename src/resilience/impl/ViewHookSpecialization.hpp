#ifndef __VIEW_HOOK_SPECIALIZATION_
#define __VIEW_HOOK_SPECIALIZATION_


#include <Kokkos_Core.hpp>
#include <Kokkos_ViewHooks.hpp>

namespace Kokkos {
namespace Impl {

template <class ViewType>
class ViewHookSpecialization<
    ViewType, typename std::enable_if<
                  (std::is_const<ViewType>::value &&
                   !std::is_same<Kokkos::AnonymousSpace,
                                 typename ViewType::memory_space>::value),
                  void>::type> {
 public:
  using view_type = ViewType;

  static inline void update_view(ViewType &view, const void *src_rec) {}

  // can copy from const view, not too
  static void deep_copy(unsigned char *buff, view_type &view) {
    using memory_space = typename view_type::memory_space;
    using exec_space   = typename memory_space::execution_space;
    Kokkos::Impl::DeepCopy<Kokkos::HostSpace, memory_space, exec_space>(
        buff, view.data(),
        view.span() * sizeof(typename view_type::value_type));
  }

  static void deep_copy(view_type &, unsigned char *) {}

  static constexpr const char *m_name = "ConstImpl";
};

template <class ViewType>
class ViewHookSpecialization<
    ViewType, typename std::enable_if<
                  (!std::is_const<ViewType>::value &&
                   !std::is_same<Kokkos::AnonymousSpace,
                                 typename ViewType::memory_space>::value && 
                    std::is_same<typename ViewType::memory_space::resilient_space,
                                 typename ViewType::memory_space>::value),
                  void>::type> {
 public:
  using view_type = ViewType;
  static inline void update_view(view_type &view, const void *src_rec) {
    using mem_space    = typename view_type::memory_space;
    using exec_space   = typename mem_space::execution_space;
    using view_traits  = typename view_type::traits;
    using value_type   = typename view_traits::value_type;
    using pointer_type = typename view_traits::value_type *;
    using handle_type =
        typename Kokkos::Impl::ViewDataHandle<view_traits>::handle_type;
    using functor_type = Kokkos::Impl::ViewValueFunctor<exec_space, value_type>;
    using header_type = Kokkos::Impl::SharedAllocationHeader;
    using record_type =
        Kokkos::Impl::SharedAllocationRecord<mem_space, functor_type>;
 
    record_type* orig_rec = (record_type*)(src_rec); // get the original ptr
 
    std::string label = orig_rec->get_label();

    record_type* const record =
      record_type::allocate(mem_space(), label, orig_rec->size());
    
    // need to assign the new record to the view map / handle
    view.assign_data_handle(
        handle_type(reinterpret_cast<pointer_type>(record->data())));

    // have to attach the destructor, but don't construct / initialize
    record->m_destroy = functor_type(
        exec_space(), (value_type *)view.impl_map().m_impl_handle, view.span());

    // This should disconnect the duplicate view from the original record and
    // attach the duplicated data to the tracker 
    view.assign_record(record);

    // add records and types to the duplicate list
    KokkosResilience::template track_duplicate<typename ViewType::traits::value_type,
                                               mem_space>(orig_rec, record);

  }

  static void deep_copy(unsigned char *buff, view_type &view) {
    using memory_space = typename view_type::memory_space;
    using exec_space   = typename memory_space::execution_space;
    Kokkos::Impl::DeepCopy<Kokkos::HostSpace, memory_space, exec_space>(
        buff, view.data(),
        view.span() * sizeof(typename view_type::value_type));
  }
  static void deep_copy(view_type &view, unsigned char *buff) {
    using memory_space = typename view_type::memory_space;
    using exec_space   = typename memory_space::execution_space;
    Kokkos::Impl::DeepCopy<Kokkos::HostSpace, memory_space, exec_space>(
        view.data(), buff,
        view.span() * sizeof(typename view_type::value_type));
  }
  static constexpr const char *m_name = "Non-ConstImpl";
};

}  // namespace Impl
}  // namespace Kokkos

#endif // __VIEW_HOOK_SPECIALIZATION_
