//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#
#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>
using namespace aie;



class layer2_0_down {
private:
    int m_xoff; 
public:
    layer2_0_down(int xoff);

  // Run:
  void run(
    adf::input_buffer<bfloat16, adf::extents<384>>& wbuf,
    input_stream<bfloat16>*                         sin ,
    adf::output_buffer<bfloat16, adf::extents<4096>>& buf);

  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( layer2_0_down::run );
  }
};

