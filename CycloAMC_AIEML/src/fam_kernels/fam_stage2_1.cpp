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

#include "fam_stage2_1.h"
#include "fam_funcs.h"
#include "parameters.h"

//------------------------------------------------------------
fam_stage2_1::fam_stage2_1(int xoff) : m_xoff(xoff) 
{
    aie::set_rounding(aie::rounding_mode::symmetric_inf);
    aie::set_saturation(aie::saturation_mode::saturate);
}

//------------------------------------------------------------
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
void fam_stage2_1::run(
    input_stream<cbfloat16>*                              sin,
    adf::input_buffer<bfloat16, adf::extents<128>>& __restrict k_2_out,
    adf::output_buffer<bfloat16, adf::extents<256>>& __restrict bout )
{
    using vec32 = aie::vector<cbfloat16, 32>;

    const int pick_idx = m_xoff - 1;
    vec32 input0,input1;

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
    }
    constexpr bfloat16 ALPHA = 2.57639e-07;
    constexpr bfloat16 BETA  = -0.306756;

    bfloat16* outptr0 = bout.data();
    bfloat16* outptr1 = outptr0 + 64;
    bfloat16* outptr2 = outptr0 + 128;

    alignas(aie::vector_decl_align) cbfloat16 xbuf0[32];
    alignas(aie::vector_decl_align) cbfloat16 ybuf0[32];
    alignas(aie::vector_decl_align) cbfloat16 xbuf1[32];
    alignas(aie::vector_decl_align) cbfloat16 ybuf1[32];

    for (int i = 0; i < 64; i++)  
    chess_prepare_for_pipelining chess_loop_range(1,)  
    {   vec32 conjugate = readincr_v<32>(sin);  
        vec32 v_in0  = aie::mul(input0,  aie::conj(conjugate));
        vec32 v_in1  = aie::mul(input1,  aie::conj(conjugate));
        aie::store_v(xbuf0, v_in0);
        aie::store_v(xbuf1, v_in1);
        fft32_cfloat(xbuf0, ybuf0);
        fft32_cfloat(xbuf1, ybuf1);
        
        *outptr0++ = aie::add(
                   aie::mul(aie::abs_square(ybuf0[8])
                           , ALPHA)
                 , BETA);
        *outptr1++ = aie::add(
                    aie::mul(aie::abs_square(ybuf1[8])
                            , ALPHA)
                  , BETA);
    }

    {
        auto kptr = aie::begin_restrict_vector<32>(k_2_out);
        auto wptr = aie::begin_restrict_vector<32>(outptr2);

        for (int i = 0; i < 4; ++i)
            *wptr++ = *kptr++;
    }
}
