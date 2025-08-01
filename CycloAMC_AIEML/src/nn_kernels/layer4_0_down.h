//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#
#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>
using namespace aie;



class layer4_0_down {
private:
    int m_xoff; 
public:
    layer4_0_down(int xoff);

  // Run:
  void run(
        adf::input_buffer<bfloat16, adf::extents<1152>>& wbuf,
        input_stream<bfloat16>*                         sin ,
        adf::output_buffer<bfloat16, adf::extents<512>>& buf);

  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( layer4_0_down::run );
  }
};

