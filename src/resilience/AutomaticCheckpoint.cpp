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
#include "AutomaticCheckpoint.hpp"
#include <algorithm>

namespace KokkosResilience
{
  namespace Detail
  {
      std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > views;
      bool iter_is_unfiltered;
      
      bool viewHolderEQ(std::unique_ptr<Kokkos::ViewHolderBase> &a, std::unique_ptr<Kokkos::ViewHolderBase> &b){
        bool labelsMatch = a->label() == b->label();
        bool pointersMatch = a->data() == b->data();
        
        fprintf(stderr, "Comparing \"%s\" vs \"%s\"\n", a->label().c_str(), b->label().c_str());
        
        //Don't remove duplicates to different pointers here, do it later so we can remove both.
        return labelsMatch && pointersMatch;
      }
        
      bool viewHolderLT(std::unique_ptr<Kokkos::ViewHolderBase> &a, std::unique_ptr<Kokkos::ViewHolderBase> &b){
        return a->label() < b->label();
      }
    
      void removeDuplicateViews(std::vector< std::unique_ptr<Kokkos::ViewHolderBase>> &viewVec){
        std::sort(viewVec.begin(), viewVec.end(), viewHolderLT);
        viewVec.erase(std::unique(viewVec.begin(), viewVec.end(), viewHolderEQ), viewVec.end());
        
        //Remove all copies of views matching labels but with different pointers
        for(int i = 0; i < viewVec.size()-1; ++i){
            int nDups = 0;
            fprintf(stderr, "Checking %d (%p) vs %d (%p) w/ size %d\n", i, viewVec[i].get(), i+1, viewVec[i+1].get(), viewVec.size());
            fprintf(stderr, "Checking \"%s\" vs \"%s\"\n", viewVec[i]->label().c_str(), viewVec[i+1]->label().c_str());
            while(  (i+nDups+1 < viewVec.size()) &&
                    (viewVec[i]->label() == viewVec[i+1+nDups]->label())){
                nDups++;
                fprintf(stderr, "Match! checking %d vs %d (%p)\n", i, i+1+nDups, viewVec[i+1+nDups].get());
                fprintf(stderr, "Checking \"%s\" vs \"%s\"\n", viewVec[i]->label().c_str(), viewVec[i+1+nDups]->label().c_str());
            }

            if(nDups > 0){
                nDups++; //Count the original too
                std::cerr << "Warning: Found " << nDups << " views w/ same label (\"" << viewVec[i]->label() << "\") and different pointers in same context!" << std::endl;
                viewVec.erase(viewVec.begin()+i, viewVec.begin()+i+nDups);
                --i;
            }
        }
      }

  }
}
