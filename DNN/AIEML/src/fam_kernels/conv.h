//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#
#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>
using namespace aie;



class conv {
private:
    int m_xoff; 
public:
    conv(int xoff);

  // Run:
  void run( input_stream<cbfloat16>*                              sin,
          adf::output_buffer<cbfloat16, adf::extents<512>>& __restrict bout );

  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( conv::run );
  }
};

