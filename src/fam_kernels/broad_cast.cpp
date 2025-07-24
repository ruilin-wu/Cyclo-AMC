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

#include "broad_cast.h"
#include "fam_funcs.h"
//#include "parameters.h"


// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

broad_cast::broad_cast(int xoff) : m_xoff(xoff) 
{
    aie::set_rounding(aie::rounding_mode::symmetric_inf);
    aie::set_saturation(aie::saturation_mode::saturate);
}



void broad_cast::run(
    input_buffer <cbfloat16, adf::extents<512>>& __restrict in0,
    input_buffer <cbfloat16, adf::extents<512>>& __restrict in1,
    input_buffer <cbfloat16, adf::extents<512>>& __restrict in2,
    input_buffer <cbfloat16, adf::extents<512>>& __restrict in3,
    output_stream<cbfloat16>* __restrict           out      )   // ★ 指针
{
    using vec8 = aie::vector<cbfloat16, 8>;

   
    //constexpr int START_PACK   = 95;   // 0..11 为左边 96 列
    constexpr int CENTER_PACKS = 64;   // 96..159 共 64×vec8 = 512 元素
    


    for (int rep = 0; rep < 1; ++rep)
    {
        const vec8* __restrict p0 =
            reinterpret_cast<const vec8*>(in0.data());
        const vec8* __restrict p1 =
            reinterpret_cast<const vec8*>(in1.data());
        const vec8* __restrict p2 =
            reinterpret_cast<const vec8*>(in2.data());
        const vec8* __restrict p3 =
            reinterpret_cast<const vec8*>(in3.data());

        for (int blk = 0; blk < CENTER_PACKS; ++blk)
        chess_prepare_for_pipelining
        chess_loop_range(1,)
        {
            writeincr(out, *p0++);   // ★ 传指针
            writeincr(out, *p1++);
            writeincr(out, *p2++);
            writeincr(out, *p3++);
        }
    }

    for (int rep = 0; rep < 1; ++rep)
    {
        const vec8* __restrict p0 =
            reinterpret_cast<const vec8*>(in0.data());
        const vec8* __restrict p1 =
            reinterpret_cast<const vec8*>(in1.data());
        const vec8* __restrict p2 =
            reinterpret_cast<const vec8*>(in2.data());
        const vec8* __restrict p3 =
            reinterpret_cast<const vec8*>(in3.data());

        for (int blk = 0; blk < CENTER_PACKS; ++blk)
        chess_prepare_for_pipelining
        chess_loop_range(1,)
        {
            writeincr(out, *p0++);   // ★ 传指针
            writeincr(out, *p1++);
            writeincr(out, *p2++);
            writeincr(out, *p3++);
        }
    }

    

    
}