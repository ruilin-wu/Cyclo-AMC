//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#
#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>
using namespace aie;



class layer0_stem_maxpool {
private:
    int m_xoff; 
public:
    layer0_stem_maxpool(int xoff);

  // Run:
  void run(input_stream<bfloat16>*  sin,
                              adf::output_buffer<bfloat16,
                                  adf::extents<8192>>& __restrict out_buf);

  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( layer0_stem_maxpool::run );
  }
};

