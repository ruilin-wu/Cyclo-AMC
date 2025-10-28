//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#
#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>
using namespace aie;



class fam_stage2_1 {
private:
    int m_xoff; 
public:
    fam_stage2_1(int xoff);

  // Run:
  void run(input_stream<cbfloat16>*                              sin,
    adf::input_buffer<bfloat16, adf::extents<128>>& __restrict k_2_out,
    adf::output_buffer<bfloat16, adf::extents<256>>& __restrict bout );

  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( fam_stage2_1::run );
  }
};

