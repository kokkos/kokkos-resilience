#ifndef INC_RESILIENCE_VIEW_HOLDER_HPP
#define INC_RESILIENCE_VIEW_HOLDER_HPP

#include <string>

namespace KokkosResilience
{
  class ViewHolderBase
  {
  public:
    
    virtual std::string get_label() const noexcept = 0;
    virtual void deep_copy_to_file_view( const std::string &fname ) = 0;
    virtual void get_contiguous_extent( char *&bytes, std::size_t &len ) = 0;
  
  private:
  };
  
  template< typename View >
  class ViewHolder : public ViewHolderBase
  {
  public:
    
    explicit ViewHolder( View &view )
      : m_view( &view )
    {}
    
    std::string get_label() const noexcept override { return m_view->label(); }
    
    void get_contiguous_extent( char *&bytes, std::size_t &len ) override
    {
      bytes = reinterpret_cast< char * >( m_view->data() );
      len = m_view->span() * sizeof( typename View::data_type );
    }
  
  private:
    
    View *m_view;
  };
}

#endif  // INC_RESILIENCE_VIEW_HOLDER_HPP
