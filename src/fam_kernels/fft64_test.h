//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#
#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>
using namespace aie;



class fft64_test {
private:
    int m_xoff; 
public:
    fft64_test(int xoff);

  // Run:
  void run( input_stream<cbfloat16>*                              sin,
          adf::output_buffer<cbfloat16, adf::extents<64>>& __restrict bout );

  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( fft64_test::run );
  }
};

