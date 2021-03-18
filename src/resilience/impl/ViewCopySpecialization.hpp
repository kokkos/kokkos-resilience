#ifndef __VIEW_COPY_SPECIALIZATION_
#define __VIEW_COPY_SPECIALIZATION_

#include <Kokkos_Core.hpp>

namespace KokkosResilience {

template <class DataType, class... Properties>
Kokkos::View<typename Kokkos::ViewTraits<DataType, Properties...>::non_const_data_type,
     typename Kokkos::ViewTraits<DataType, Properties...>::array_layout, Kokkos::HostSpace,
     Kokkos::MemoryTraits<Kokkos::Unmanaged> >
make_unmanaged_host_view(const Kokkos::View<DataType, Properties...> &view,
                         unsigned char *buff) {
  using traits_type   = Kokkos::ViewTraits<DataType, Properties...>;
  using new_data_type = typename traits_type::non_const_data_type;
  using layout_type   = typename traits_type::array_layout;
  using new_view_type =
      Kokkos::View<new_data_type, layout_type, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  return new_view_type(
      reinterpret_cast<typename new_view_type::pointer_type>(buff),
      view.rank_dynamic > 0 ? view.extent(0) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 1 ? view.extent(1) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 2 ? view.extent(2) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 3 ? view.extent(3) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 4 ? view.extent(4) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 5 ? view.extent(5) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 6 ? view.extent(6) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 7 ? view.extent(7) : KOKKOS_IMPL_CTOR_DEFAULT_ARG);
}

} // namespace KokkosResilience

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class ViewType>
class ViewHookCopyView<
    ViewType, typename std::enable_if<
                  (std::is_const<ViewType>::value &&
                   !std::is_same<Kokkos::AnonymousSpace,
                                 typename ViewType::memory_space>::value),
                  void>::type> {
 public:
  using view_type = ViewType;

  static inline void copy_view(ViewType &, const void *) {}
  static inline void copy_view(const void *, ViewType &) {}
  static constexpr const char *m_name = "ConstImpl";
};

template <class ViewType>
class ViewHookCopyView<
    ViewType, typename std::enable_if<
                  (!std::is_const<ViewType>::value &&
                   !std::is_same<Kokkos::AnonymousSpace,
                                 typename ViewType::memory_space>::value && 
                    std::is_same<typename ViewType::memory_space::resilient_space,
                                 typename ViewType::memory_space>::value),
                  void>::type> {
 public:
  static inline void copy_view(ViewType & view, unsigned char * ptr) {
     auto src = KokkosResilience::make_unmanaged_host_view( view, ptr );
     Kokkos::deep_copy( view, src );
  }

  static inline void copy_view(unsigned char* ptr, ViewType & view) {
     auto src = KokkosResilience::make_unmanaged_host_view( view, ptr );
     Kokkos::deep_copy( src, view );
  }

  static constexpr const char *m_name = "Non-ConstImpl";
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif // __VIEW_COPY_SPECIALIZATION_
