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
#ifndef INC_RESILIENCE_DYNAMICVIEWHOOKS_HPP
#define INC_RESILIENCE_DYNAMICVIEWHOOKS_HPP

#include "ViewHolder.hpp"
#include <Kokkos_Core.hpp>
#include <mutex>
#include <functional>

namespace KokkosResilience {

struct DynamicViewHooks {
  using callback_type       = std::function<void(const ViewHolder &)>;
  using const_callback_type = std::function<void(const ConstViewHolder &)>;

  class callback_overload_set {
   public:
    template <class DataType, class... Properties>
    void call(Kokkos::View<DataType, Properties...> &view) {
      std::lock_guard<std::mutex> lock(m_mutex);

      if (!any_set()) return;

      auto holder = make_view_holder(view);
      do_call(std::move(holder));
    }

    template <typename F>
    void set_callback(F &&cb) {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_callback = std::forward<F>(cb);
    }

    template <typename F>
    void set_const_callback(F &&cb) {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_const_callback = std::forward<F>(cb);
    }

    void clear_callback() {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_callback = {};
    }

    void clear_const_callback() {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_const_callback = {};
    }

    void reset() {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_callback       = {};
      m_const_callback = {};
    }

   private:
    // Not thread safe, don't call outside of mutex lock
    void do_call(const ViewHolder &view) {
      if (m_callback) m_callback(view);
    }

    // Not thread safe, don't call outside of mutex lock
    void do_call(const ConstViewHolder &view) {
      if (m_const_callback) m_const_callback(view);
    }

    // Not thread safe, call inside of mutex
    bool any_set() const noexcept {
      return static_cast<bool>(m_callback) ||
             static_cast<bool>(m_const_callback);
    }

    callback_type m_callback;
    const_callback_type m_const_callback;
    std::mutex m_mutex;
  };

  static void reset() {
    copy_constructor_set.reset();
    copy_assignment_set.reset();
    move_constructor_set.reset();
    move_assignment_set.reset();
  }

  static callback_overload_set copy_constructor_set;
  static callback_overload_set copy_assignment_set;
  static callback_overload_set move_constructor_set;
  static callback_overload_set move_assignment_set;

  /**
   * Mark whether or not we are currently in a dynamic view hook.
   * Because ViewHolder itself copies a view, we need to suppress
   * recursively entering the view hooks while we area already handling one.
   *
   * This is thread_local because it should only apply to a specific thread;
   * that is, we don't care about multiple threads calling the dynamic view
   * hook. It's the user's responsibility to make sure that the hook
   * is thread-safe.
   */
  static thread_local bool reentrant;
};

namespace Impl {
template <class ViewType, class Traits = typename ViewType::traits,
          class Enabled = void>
struct DynamicViewHooksCaller;

template <class ViewType, class Traits, class Enabled>
struct DynamicViewHooksCaller {
  static void call_construct_hooks(ViewType &) {}
  static void call_copy_construct_hooks(ViewType &) {}
  static void call_copy_assign_hooks(ViewType &) {}
  static void call_move_construct_hooks(ViewType &) {}
  static void call_move_assign_hooks(ViewType &) {}
};

template <class ViewType, class Traits>
struct DynamicViewHooksCaller<
    ViewType, Traits,
    typename std::enable_if<!std::is_same<typename Traits::memory_space,
                                          Kokkos::AnonymousSpace>::value>::type> {
  static void call_copy_construct_hooks(ViewType &view) {
    if (!DynamicViewHooks::reentrant) {
      DynamicViewHooks::reentrant = true;
      DynamicViewHooks::copy_constructor_set.call(view);
      DynamicViewHooks::reentrant = false;
    }
  }

  static void call_copy_assign_hooks(ViewType &view) {
    if (!DynamicViewHooks::reentrant) {
      DynamicViewHooks::reentrant = true;
      DynamicViewHooks::copy_assignment_set.call(view);
      DynamicViewHooks::reentrant = false;
    }
  }

  static void call_move_construct_hooks(ViewType &view) {
    if (!DynamicViewHooks::reentrant) {
      DynamicViewHooks::reentrant = true;
      DynamicViewHooks::move_constructor_set.call(view);
      DynamicViewHooks::reentrant = false;
    }
  }

  static void call_move_assign_hooks(ViewType &view) {
    if (!DynamicViewHooks::reentrant) {
      DynamicViewHooks::reentrant = true;
      DynamicViewHooks::move_assignment_set.call(view);
      DynamicViewHooks::reentrant = false;
    }
  }
};
}  // namespace Impl

struct DynamicViewHooksSubscriber {
  template <typename View>
  static void copy_constructed(View &self, const View &) {
    Impl::DynamicViewHooksCaller<View>::call_copy_construct_hooks(self);
  }
  template <typename View>
  static void copy_assigned(View &self, const View &) {
    Impl::DynamicViewHooksCaller<View>::call_copy_assign_hooks(self);
  }

  template <typename View>
  static void move_constructed(View &self, const View &) {
    Impl::DynamicViewHooksCaller<View>::call_move_construct_hooks(self);
  }
  template <typename View>
  static void move_assigned(View &self, const View &) {
    Impl::DynamicViewHooksCaller<View>::call_move_assign_hooks(self);
  }
};

template< typename... Args >
using View = Kokkos::View< Args..., Kokkos::Experimental::SubscribableViewHooks< KokkosResilience::DynamicViewHooksSubscriber > >;
}  // namespace KokkosResilience

#endif  // INC_RESILIENCE_DYNAMICVIEWHOOKS_HPP
