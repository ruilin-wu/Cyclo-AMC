//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#
#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>
using namespace aie;



class fam_stage1 {
private:
    int m_xoff; 
public:
    fam_stage1(int xoff);

  // Run:
  void run(
    adf::input_buffer <cbfloat16, adf::extents<176>>  & __restrict inputx0,
    adf::output_buffer<cbfloat16, adf::extents<512>> & __restrict outputy );

  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( fam_stage1::run );
  }
};

