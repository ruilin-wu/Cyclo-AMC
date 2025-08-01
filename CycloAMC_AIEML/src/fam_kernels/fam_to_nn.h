//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#
#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>
using namespace adf;



class fam_to_nn {
private:
    int m_xoff; 
public:
    fam_to_nn(int xoff);

  // Run:
    void run( input_stream <cbfloat16>* __restrict in0,
    output_stream<cbfloat16>* __restrict out );


  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( fam_to_nn::run );
  }
};

