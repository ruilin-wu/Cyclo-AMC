//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#
#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>
using namespace aie;



class layer0_stem {
private:
    int m_xoff; 
public:
    layer0_stem(int xoff);

  // Run:
  void run( adf::input_buffer<bfloat16, adf::extents<72>>& wbuf,
                       input_stream<bfloat16>*                        sin ,
                       /*★ 改 1：缓冲 → 流 */                       
                       output_stream<bfloat16>*                       bout );

  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( layer0_stem::run );
  }
};

