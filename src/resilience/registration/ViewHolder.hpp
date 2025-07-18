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

#ifndef _INC_RESILIENCE_REGISTRATION_VIEWHOLDER_HPP
#define _INC_RESILIENCE_REGISTRATION_VIEWHOLDER_HPP

#include "Registration.hpp"
#include "resilience/view_hooks/ViewHolder.hpp"
#include "resilience/context/Context.hpp"

namespace KokkosResilience::Impl::Registration {
  class ViewHolder : public Base {
  public:
    ViewHolder() = delete;

    ViewHolder(
      ContextBase& ctx,
      const KokkosResilience::ViewHolder& view,
      const std::string& label = ""
    ) : Base(view->label()), m_view(view), m_ctx(ctx) {};

    const serializer_t serializer() const override{
      return [&, this](std::ostream& stream){
        size_t buffer_size = 
          need_buffer ? m_view->data_type_size()*m_view->size() : 0;
        char* buf = m_ctx.get_scratch_buffer(buffer_size);
    
        m_view->serialize(stream, buf);
        return stream.good();
      };
    }

    const deserializer_t deserializer() const override{
      return [&, this](std::istream& stream){
        size_t buffer_size = 
          need_buffer ? m_view->data_type_size()*m_view->size() : 0;
        char* buf = m_ctx.get_scratch_buffer(buffer_size);
    
        m_view->deserialize(stream, buf);
        return stream.good();
      };
    }

    const bool is_same_reference(const Registration& other_reg) const override{
      auto other = dynamic_cast<ViewHolder*>(other_reg.get());
      
      if(!other){
        fprintf(stderr,
          "KokkosResilience: Warning, member name %s is shared by more than 1"
          " registration type\n", name.c_str()
        );
        return false;
      }
    
      // This currently assumes the two views are equal or subviews (ie no name
      //  collisions), and that a larger data() pointer implies a subview (ie
      //  we can deal well with subviews of subviews, but not two different
      //  subviews of the same view).
      // Should probably be updated once we have the new view hooking system
      return m_view->data() <= other->m_view->data();
    }

  private:
    const KokkosResilience::ViewHolder m_view;

    const bool need_buffer = 
    #ifdef KR_ENABLE_MAGISTRATE
        false;
    #else
        !(m_view->span_is_contiguous() && m_view->is_host_space());
    #endif

    ContextBase& m_ctx;
  };

  template<>
  struct Factory<const KokkosResilience::ViewHolder>
   : public Factory<const KokkosResilience::ViewHolder, ViewHolder>{};
}

#endif
