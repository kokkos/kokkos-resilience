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
