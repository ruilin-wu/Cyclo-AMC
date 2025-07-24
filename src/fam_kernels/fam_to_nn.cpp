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

#include "fam_to_nn.h"
#include "fam_funcs.h"
//#include "parameters.h"


// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

fam_to_nn::fam_to_nn(int xoff) : m_xoff(xoff) 
{
    aie::set_rounding(aie::rounding_mode::symmetric_inf);
    aie::set_saturation(aie::saturation_mode::saturate);
}


/**/
#include <aie_api/aie.hpp>

#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>

void fam_to_nn::run(
    input_stream <cbfloat16>* __restrict in0,
    output_stream<cbfloat16>* __restrict out )          // ★ 指针
{
    using vec32 = aie::vector<cbfloat16, 32>;

    constexpr int VEC_PER_FRAME = 16384*4 / 32;           // 512 ← 每帧 vec32 数

    /*──── 逐 vec32 转发 ────*/
    for (int i = 0; i < VEC_PER_FRAME; ++i)
    chess_prepare_for_pipelining
    chess_loop_range(1,)
    {
        vec32 v = readincr_v<32>(in0);                  // 从 AIE-stream 读取
        writeincr(out, v);                              // 写回 AIE-stream
    }
}
