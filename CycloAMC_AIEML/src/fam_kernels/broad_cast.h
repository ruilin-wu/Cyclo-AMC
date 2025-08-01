//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#
#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>
using namespace adf;



class broad_cast {
private:
    int m_xoff; 
public:
    broad_cast(int xoff);

  // Run:
    void run( input_buffer <cbfloat16, adf::extents<512>>& __restrict in0,
              input_buffer <cbfloat16, adf::extents<512>>& __restrict in1,
              input_buffer <cbfloat16, adf::extents<512>>& __restrict in2,
              input_buffer <cbfloat16, adf::extents<512>>& __restrict in3,
              output_stream<cbfloat16>* __restrict           out      );


  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( broad_cast::run );
  }
};

