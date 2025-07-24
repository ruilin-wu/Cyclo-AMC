//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#
#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>
using namespace aie;



class layer4_1_identity {
private:
    int m_xoff; 
public:
    layer4_1_identity(int xoff);

  // Run:
  void run(
    adf::input_buffer<bfloat16, adf::extents<20>>& wbuf,
    input_stream<bfloat16>*                                   sin,
            adf::output_buffer<bfloat16, adf::extents<512>>&         buf);

  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( layer4_1_identity::run );
  }
};

