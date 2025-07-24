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

#include "fam_stage2.h"
#include "fam_funcs.h"
#include "parameters.h"


// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

fam_stage2::fam_stage2(int xoff) : m_xoff(xoff) 
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
void fam_stage2::run(
    input_stream<cbfloat16>*                              sin,
    adf::output_buffer<bfloat16, adf::extents<128>>& __restrict bout )
{
    using vec32 = aie::vector<cbfloat16, 32>;

    const int pick_idx = m_xoff - 1;  // m_xoff = 1…64 → pick_idx = 0…63
    vec32 input0,input1;                      // 用于保存选中的向量

    for (int blk = 0; blk < 64; ++blk)
    chess_prepare_for_pipelining
    chess_loop_range(64,)
    {
        vec32 vin = readincr_v<32>(sin);

        if (blk == pick_idx) {
            input0 = vin; // 保留为 input0
        }
        if (blk == pick_idx+1) {
            input1 = vin; // 保留为 input0
        }
    }
    constexpr bfloat16 ALPHA = 6.93232e-05;   // 1/(std_train+1e-5)
    constexpr bfloat16 BETA  = -0.296705;    // –mean_train*ALPHA

    // 写入 output buffer，只写一次
    bfloat16* outptr0 = bout.data();
    bfloat16* outptr1 = outptr0 + 64;        // [64 … 127]
    //*reinterpret_cast<vec32*>(outptr) = input0;

    alignas(aie::vector_decl_align) cbfloat16 xbuf0[32];    // FFT 输入
    alignas(aie::vector_decl_align) cbfloat16 ybuf0[32];    // FFT 输出 (临时)
    alignas(aie::vector_decl_align) cbfloat16 xbuf1[32];    // FFT 输入
    alignas(aie::vector_decl_align) cbfloat16 ybuf1[32];    // FFT 输出 (临时)

    for (int i = 0; i < 64; i++)  
    chess_prepare_for_pipelining chess_loop_range(1,)  
    {   vec32 conjugate = readincr_v<32>(sin);  
        vec32 v_in0  = aie::mul(input0,  aie::conj(conjugate));        // your temp1
        vec32 v_in1  = aie::mul(input1,  aie::conj(conjugate));        // your temp1
        //*itw++  = v_in;
        aie::store_v(xbuf0, v_in0);                                // ① 存入本地数组        
        aie::store_v(xbuf1, v_in1);                                // ① 存入本地数组
        fft32_cfloat(xbuf0, ybuf0);                                    // ② 调 FFT
        fft32_cfloat(xbuf1, ybuf1);                                    // ② 调 FFT
        //*outptr++ = aie::abs_square(ybuf[8]); 
        
        *outptr0++ = aie::add(                     // ② + β
                   aie::mul(aie::abs_square(ybuf0[8]) // ① |z|² × α
                           , ALPHA)            // α
                 , BETA);
        *outptr1++ = aie::add(                     // ② + β
                    aie::mul(aie::abs_square(ybuf1[8]) // ① |z|² × α
                            , ALPHA)            // α
                  , BETA);                       // β
        
    }
}
