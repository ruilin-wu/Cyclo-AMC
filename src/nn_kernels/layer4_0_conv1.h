//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#
#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>
using namespace aie;



class layer4_0_conv1 {
private:
    int m_xoff; 
public:
    layer4_0_conv1(int xoff);

  // Run:
  void run(
        adf::input_buffer<bfloat16, adf::extents<10368>>& wbuf,
        input_stream<bfloat16>*                          sin ,
        output_stream<bfloat16>*                         sout);

  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( layer4_0_conv1::run );
  }
};

