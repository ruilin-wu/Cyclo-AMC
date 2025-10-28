#pragma once
#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>




class layer5_head {
private:
    int m_xoff; 
public:
    layer5_head(int xoff);

  // Run:
  void run(adf::input_buffer<bfloat16, adf::extents<776>>& w_lin_b,
                      input_stream<bfloat16>*                        ifm_s,
                      output_stream<bfloat16>*                       ofm_s);

  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( layer5_head::run );
  }
};