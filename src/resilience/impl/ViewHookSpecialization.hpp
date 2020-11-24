#ifndef __VIEW_HOOK_SPECIALIZATION_
#define __VIEW_HOOK_SPECIALIZATION_


#include <Kokkos_Core.hpp>

namespace Kokkos {
namespace Experimental {

template <class ViewAttorneyType>
class ViewHookUpdate<
    ViewAttorneyType, typename std::enable_if<
                  (std::is_const<typename ViewAttorneyType::view_type>::value ||
                   std::is_same<Kokkos::AnonymousSpace,
                                 typename ViewAttorneyType::view_type::memory_space>::value || 
                   !std::is_same<typename ViewAttorneyType::view_type::memory_space::resilient_space,
                                 typename ViewAttorneyType::view_type::memory_space>::value),
                  void>::type> {
 public:
  using view_att_type = ViewAttorneyType;

  static inline void update_view(view_att_type &) {}
  static constexpr const char *m_name = "ConstImpl";
};

template <class ViewAttorneyType>
class ViewHookUpdate<
    ViewAttorneyType, typename std::enable_if<
                  !(std::is_const<typename ViewAttorneyType::view_type>::value ||
                   std::is_same<Kokkos::AnonymousSpace,
                                 typename ViewAttorneyType::view_type::memory_space>::value || 
                   !std::is_same<typename ViewAttorneyType::view_type::memory_space::resilient_space,
                                 typename ViewAttorneyType::view_type::memory_space>::value),
                  void>::type> {
 public:
  using view_att_type   = ViewAttorneyType;
  using view_type       = typename view_att_type::view_type;
  static inline void update_view(view_att_type &view) {
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
 
    record_type *orig_rec =
        (record_type *)(view.rec_ptr());  // get the original record
 
    std::string label = orig_rec->get_label();

    record_type* const record =
      record_type::allocate(mem_space(), label, orig_rec->size());
    
    // need to assign the new record to the view map / handle
    view.update_data_handle(
        handle_type(reinterpret_cast<pointer_type>(record->data())));

    record->m_destroy = functor_type(
        exec_space(), reinterpret_cast<value_type *>(record->data()),
        view.get_view().span(), label);

    // This should disconnect the duplicate view from the original record and
    // attach the duplicated data to the tracker
    view.assign_view_record(record);

    // add records and types to the duplicate list
    KokkosResilience::template track_duplicate<value_type,
                                               mem_space>(orig_rec, record);

  }

  static constexpr const char *m_name = "Non-ConstImpl";
};

}  // namespace Experimental
}  // namespace Kokkos

#endif // __VIEW_HOOK_SPECIALIZATION_
