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

#include "fam_stage1.h"
#include "fam_funcs.h"
//#include "parameters.h"


// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

fam_stage1::fam_stage1(int xoff) : m_xoff(xoff) 
{
    aie::set_rounding(aie::rounding_mode::symmetric_inf);
    aie::set_saturation(aie::saturation_mode::saturate);
}


/**/
void fam_stage1::run(
    adf::input_buffer <cbfloat16, adf::extents<176>>  & __restrict inputx0,
    adf::output_buffer<cbfloat16, adf::extents<512>> & __restrict outputy )
{
    /* === 基本指针 === */
    cbfloat16*       __restrict xin  = inputx0.data();     // 704 × cbfloat16
    cbfloat16*       __restrict yout = outputy.data();     // 2048 × cbfloat16
    const bfloat16*  __restrict win  = window_factor_64;     // 256 × bf16


    alignas(aie::vector_decl_align) cbfloat16 tmp1[64];
    alignas(aie::vector_decl_align) cbfloat16 tmp2[64];
    alignas(aie::vector_decl_align) cbfloat16 tmp3[64];



    for (int seg = 0; seg < 8; ++seg)
    chess_prepare_for_pipelining
    chess_loop_range(1,)
    {   
        int transpose_idx = seg & 3;
        cbfloat16* in_seg  = xin  + seg *  16;   // 64-点滑窗
        cbfloat16* out_seg = yout ;   // 输出写 256 点

        window_mul_cbfloat16(in_seg, win, tmp1);   // 乘窗
        fft64_cfloat(tmp1, tmp3);        // FFT + fftshift    
        stage1_dc(tmp3, transpose_idx, tmp2);
        transpose_64(tmp2, out_seg, seg);
    }
}


/*
void fam_stage1::run(
    adf::input_buffer <cbfloat16, adf::extents<176>>  & __restrict inputx0,
    adf::output_buffer<cbfloat16, adf::extents<176>> & __restrict outputy )
{

    cbfloat16*       __restrict xin  = inputx0.data();     // 704 × cbfloat16
    cbfloat16*       __restrict yout = outputy.data();     // 2048 × cbfloat16
    const bfloat16*  __restrict win  = window_factor_64;     // 256 × bf16


    alignas(aie::vector_decl_align) cbfloat16 tmp1[64];
    alignas(aie::vector_decl_align) cbfloat16 tmp2[64];
    alignas(aie::vector_decl_align) cbfloat16 tmp3[64];



    for (int seg = 0; seg < 8; ++seg)
    chess_prepare_for_pipelining
    chess_loop_range(1,)
    {   
        int transpose_idx = seg & 3;
        cbfloat16* in_seg  = xin  + seg *  16;   // 64-点滑窗
        cbfloat16* out_seg = yout;   // 输出写 256 点

        window_mul_cbfloat16(in_seg, win, out_seg);   // 乘窗
        
        
        window_mul_cbfloat16(in_seg, win, tmp1);   // 乘窗
        fft64_cfloat(tmp1, tmp2, tmp3);        // FFT + fftshift
        stage1_dc(tmp3, transpose_idx, tmp2);
        transpose_64(tmp2, out_seg, seg);
        

    }
}
*/