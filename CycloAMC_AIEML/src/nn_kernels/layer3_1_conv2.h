//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#
#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>
using namespace aie;



class layer3_1_conv2 {
private:
    int m_xoff; 
public:
    layer3_1_conv2(int xoff);

  // Run:
  void run(
        adf::input_buffer<bfloat16, adf::extents<15552>>& wbuf,
        input_stream<bfloat16>*                          sin ,
        adf::input_buffer<bfloat16, adf::extents<1536>>& residual,
        output_stream<bfloat16>*                         sout);

  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( layer3_1_conv2::run );
  }
};

