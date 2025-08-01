//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#
#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

using namespace adf;

#include "fft64_test.h"
#include "fam_funcs.h"
#include "parameters.h"


// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

fft64_test::fft64_test(int xoff) : m_xoff(xoff) 
{
    aie::set_rounding(aie::rounding_mode::symmetric_inf);
    aie::set_saturation(aie::saturation_mode::saturate);
}

// ------------------------------------------------------------
// Run
// ------------------------------------------------------------

// ① 头文件不变

//------------------------------------------------------------
// Run
//------------------------------------------------------------

#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
void fft64_test::run(
    input_stream<cbfloat16>* sin,
    adf::output_buffer<cbfloat16, adf::extents<64>>& __restrict bout)
{
    using vec64 = aie::vector<cbfloat16, 64>;

    vec64 vin = readincr_v<64>(sin);

    alignas(aie::vector_decl_align) cbfloat16 xbuf[64]; // FFT 输入
    alignas(aie::vector_decl_align) cbfloat16 ybuf[64]; // FFT 输出

    aie::store_v(xbuf, vin);            
    fft64_cfloat(xbuf, ybuf);           

    auto it = aie::begin_restrict_vector<64>(bout); 
    vec64 vy = aie::load_v<64>(ybuf); // 把数组封装成向量
    *it = vy;                
}

