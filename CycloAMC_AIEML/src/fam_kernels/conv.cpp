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

#include "conv.h"
#include "fam_funcs.h"
#include "parameters.h"

//------------------------------------------------------------
conv::conv(int xoff) : m_xoff(xoff) 
{
    aie::set_rounding(aie::rounding_mode::symmetric_inf);
    aie::set_saturation(aie::saturation_mode::saturate);
}

//------------------------------------------------------------
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>

void conv::run(
    input_buffer <cbfloat16, adf::extents<2048>>& __restrict in0,
    input_buffer <cbfloat16, adf::extents<2048>>& __restrict in1,
    input_buffer <cbfloat16, adf::extents<2048>>& __restrict in2,
    input_buffer <cbfloat16, adf::extents<2048>>& __restrict in3,
    adf::output_buffer<cbfloat16, adf::extents<512>>& __restrict bout )
{
    using vec32 = aie::vector<cbfloat16,32>;

    constexpr int KEEP_V32  = 2048 / 32;
    constexpr int FRAME_V32 = 8192 / 32;
    constexpr int DROP_V32  = FRAME_V32 - KEEP_V32;

    auto itw = aie::begin_vector<32>(bout);

    for (int i = 0; i < 512/32; ++i)
    chess_prepare_for_pipelining
    chess_loop_range(1,)
    {   
        vec32 v = readincr_v<32>(sin);
        *itw++  = v;
    }

    for (int i = 0; i < 8192/32 - 512/32; ++i)
    chess_prepare_for_pipelining
    chess_loop_range(1,)
    {
        vec32 v = readincr_v<32>(sin);
    }
}
