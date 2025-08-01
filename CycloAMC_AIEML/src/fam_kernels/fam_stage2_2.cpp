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

#include "fam_stage2_2.h"
#include "fam_funcs.h"
#include "parameters.h"

//------------------------------------------------------------
fam_stage2_2::fam_stage2_2(int xoff) : m_xoff(xoff) 
{
    aie::set_rounding(aie::rounding_mode::symmetric_inf);
    aie::set_saturation(aie::saturation_mode::saturate);
}

//------------------------------------------------------------
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
void fam_stage2_2::run(
    input_stream<cbfloat16>*                              sin,
    adf::output_buffer<bfloat16, adf::extents<256>>& __restrict bout )
{
    using vec32 = aie::vector<cbfloat16, 32>;

    const int pick_idx = m_xoff - 1;
    vec32 input0,input1,input2,input3;

    for (int blk = 0; blk < 64; ++blk)
    chess_prepare_for_pipelining
    chess_loop_range(64,)
    {
        vec32 vin = readincr_v<32>(sin);

        if (blk == pick_idx) {
            input0 = vin;
        }
        if (blk == pick_idx+1) {
            input1 = vin;
        }
        if (blk == pick_idx+2) {
            input2 = vin;
        }
        if (blk == pick_idx+3) {
            input3 = vin;
        }
    }
    constexpr bfloat16 ALPHA = 6.93232e-05;
    constexpr bfloat16 BETA  = -0.296705;

    bfloat16* outptr0 = bout.data();
    bfloat16* outptr1 = outptr0 + 64;
    bfloat16* outptr2 = outptr0 + 128;
    bfloat16* outptr3 = outptr0 + 192;

    alignas(aie::vector_decl_align) cbfloat16 xbufA[32];
    alignas(aie::vector_decl_align) cbfloat16 xbufB[32];
    alignas(aie::vector_decl_align) cbfloat16 ybufA[32];
    alignas(aie::vector_decl_align) cbfloat16 ybufB[32];

    for (int i = 0; i < 64; i++)  
    chess_prepare_for_pipelining chess_loop_range(1,)  
    {
        vec32 conjugate = readincr_v<32>(sin);

        vec32 v_in0 = aie::mul(input0, aie::conj(conjugate));
        vec32 v_in1 = aie::mul(input1, aie::conj(conjugate));
        vec32 v_in2 = aie::mul(input2, aie::conj(conjugate));
        vec32 v_in3 = aie::mul(input3, aie::conj(conjugate));

        aie::store_v(xbufA, v_in0);
        fft32_cfloat(xbufA, ybufA);
        *outptr0++ = aie::add(aie::mul(aie::abs_square(ybufA[8]), ALPHA), BETA);

        aie::store_v(xbufB, v_in1);
        fft32_cfloat(xbufB, ybufB);
        *outptr1++ = aie::add(aie::mul(aie::abs_square(ybufB[8]), ALPHA), BETA);

        aie::store_v(xbufA, v_in2);
        fft32_cfloat(xbufA, ybufA);
        *outptr2++ = aie::add(aie::mul(aie::abs_square(ybufA[8]), ALPHA), BETA);

        aie::store_v(xbufB, v_in3);
        fft32_cfloat(xbufB, ybufB);
        *outptr3++ = aie::add(aie::mul(aie::abs_square(ybufB[8]), ALPHA), BETA);
    }
}
