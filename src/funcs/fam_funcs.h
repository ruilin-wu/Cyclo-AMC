//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#ifndef __FAM_FUNCS_H__ 
#define __FAM_FUNCS_H__
#
#include <aie_api/aie.hpp>
using namespace aie;

#include <adf.h>
#include "parameters.h"
#include "fft256_cfloat_twiddles.h"




//#include "fft_twiddle_lut_dit_cbfloat.h"
/*
inline __attribute__((always_inline)) 
void window_fam (cbfloat16 * restrict px0,  cbfloat16 * restrict py0)
{   
    v8float * restrict ptw1 = (v8float * restrict) window_factor1;

    v8float * restrict pi1 = (v8float * restrict) px0;

    v8float * restrict po1 = (v8float * restrict) py0;
  
    for (int j = 0; j < 32; ++j)  
        chess_prepare_for_pipelining chess_flatten_loop
    {
        
        v8float x1 = *pi1++;
        v8float x2 = *pi1++;      
        v8float coef1 = *ptw1++;
        v8float coef2 = *ptw1++;       
        
        *po1++ = fpmul(x1, coef1);
        *po1++ = fpmul(x2, coef2);

        
    }
}

*/
inline __attribute__((always_inline))
void passthru( cbfloat16 * restrict x,cbfloat16 * restrict y)
{
  auto itr = aie::begin_vector<16>(x);
  auto itw = aie::begin_vector<16>(y);
  for (unsigned cc=0; cc < 704 /16/2; cc++) {
    *itw++ = *itr++;
    *itw++ = *itr++;
  }
}


inline __attribute__((always_inline))
void copy_segments_704_to_2048(cbfloat16* __restrict x, cbfloat16* __restrict y)
{
    /* 8 段起始偏移（单位：元素） */
    constexpr int SEG_OFF[8] = {   0,  64, 128, 192, 256, 320, 384, 448 };

    /* 每段 256 样本；每向量 16 样本 → 16 次循环 */
    for (int s = 0; s < 8; ++s) {

        /* 输入 / 输出迭代器：每次递增 16 个元素 */
        auto itr = aie::begin_vector<16>(x + SEG_OFF[s]);   // 段首
        auto itw = aie::begin_vector<16>(y + s * 256);      // y 中对应段首

        /* 256 / 16 = 16 向量搬运 */
        for (int blk = 0; blk < 16; ++blk) {
            *itw++ = *itr++;   // 一次搬 16 × cbfloat16
        }
    }
}


void window_mul_cbfloat16(const cbfloat16* __restrict input,
                          const bfloat16*  __restrict window,
                          cbfloat16*       __restrict output)
{
    // reinterpret as scalar bfloat16*
    const bfloat16* __restrict xin  = reinterpret_cast<const bfloat16*>(input);
    bfloat16*       __restrict yout = reinterpret_cast<bfloat16*>(output);

    auto it_x = aie::begin_vector<32>(xin);
    auto it_w = aie::begin_vector<32>(window);
    auto it_y = aie::begin_vector<32>(yout);

    for (int blk = 0; blk < 64 * 2 / 32; ++blk)  // 64 bfloat16 = 32 cbfloat16
    chess_prepare_for_pipelining
    chess_loop_range(1,){
        aie::vector<bfloat16, 32> vx = *it_x++;
        aie::vector<bfloat16, 32> vw = *it_w++;
        aie::vector<bfloat16, 32> vy = aie::mul(vx, vw);
        *it_y++ = vy;
    }
}


// =======================================================
// DC-scaling for 256-point segment  (64 × 4-lane vector)
// =======================================================
inline __attribute__((always_inline))
void stage1_dc(cbfloat16*       __restrict px0,   // 64×cbfloat16 输入
               int             index,           // 0–3 选系数表
               cbfloat16*       __restrict py0)   // 64×cbfloat16 输出
{
    /* -------- 1. 选 4-元素系数 -------- */
    const cbfloat16* coef_tab;
    switch(index ){
        case 0:  coef_tab = reinterpret_cast<const cbfloat16*>(dc_coef1); break;
        case 1:  coef_tab = reinterpret_cast<const cbfloat16*>(dc_coef2); break;
        case 2:  coef_tab = reinterpret_cast<const cbfloat16*>(dc_coef3); break;
        case 3:  coef_tab = reinterpret_cast<const cbfloat16*>(dc_coef4); break;
    }
    /* 常量向量：{c0,c1,c2,c3} */
    const aie::vector<cbfloat16,4> vcoef =
        *(const aie::vector<cbfloat16,4>*)coef_tab;

    /* -------- 2. 逐块(4 复数) 乘 -------- */
    auto it_in  = aie::begin_vector<4>(px0);  // 64 个块
    auto it_out = aie::begin_vector<4>(py0);

    for(int blk = 0; blk < 64/4; ++blk)         // 256 / 4 = 64
    chess_prepare_for_pipelining
    chess_loop_range(1,)
    {
        aie::vector<cbfloat16,4> vin  = *it_in++;
        aie::vector<cbfloat16,4> vout = aie::mul(vin, vcoef);
        *it_out++ = vout;
    }
}

inline __attribute__((always_inline)) void transpose_256(cbfloat16* restrict tmp2, cbfloat16* restrict ybuff, int index) {
    
    for (int i = 0; i < 256; i++) 
        chess_prepare_for_pipelining
        chess_loop_range(256,256)
    {
        ybuff[index + i*8] = tmp2[i];
    }
}

inline __attribute__((always_inline)) void transpose_64(cbfloat16* restrict tmp2, cbfloat16* restrict ybuff, int index) {
    
    for (int i = 0; i < 64; i++) 
        chess_prepare_for_pipelining
        chess_loop_range(64,64)
    {
        ybuff[index + i*8] = tmp2[i];
    }
}


inline __attribute__((always_inline))
void fft256_cfloat(cbfloat16 * restrict x,cbfloat16 * restrict tmp1, cbfloat16 * restrict y)
{
    //----------------------------------------------------------------
    // Stage-0 : Radix-4, Vec 16, 本地 64 点
    //----------------------------------------------------------------
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw0_0[1]	= TWID2560_0;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw0_1[1]	= TWID2560_1;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw0_2[1]	= TWID2560_2;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw1_0[4]	= TWID2561_0;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw1_1[4]	= TWID2561_1;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw1_2[4]	= TWID2561_2;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw2_0[16]	= TWID2562_0;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw2_1[16]	= TWID2562_1;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw2_2[16]	= TWID2562_2;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw3_0[64]	= TWID2563_0;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw3_1[64]	= TWID2563_1;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw3_2[64]	= TWID2563_2;
    constexpr unsigned shift_tw = 0;
    constexpr unsigned shift    = 0;
    constexpr bool inv      = false;
    aie::fft_dit_r4_stage<64>(x, tw0_1, tw0_0, tw0_2, 256, shift_tw, shift, inv, y);
    aie::fft_dit_r4_stage<16>(y, tw1_1, tw1_0, tw1_2, 256, shift_tw, shift, inv, x);
    aie::fft_dit_r4_stage<4>(x, tw2_1, tw2_0, tw2_2, 256, shift_tw, shift, inv, tmp1);
    aie::fft_dit_r4_stage<1>(tmp1, tw3_1, tw3_0, tw3_2, 256, shift_tw, shift, inv, y);
    //----------------------------------------------------------------
    // FFTSHIFT: swap first 128 and last 128 complex samples
    //----------------------------------------------------------------
    bfloat16* p = reinterpret_cast<bfloat16*>(y);  // 512 bfloat16s = 256 cbfloat16

    for (int i = 0; i < 128 * 2; i += 32)           // 每次交换 16 复数（32 bfloat16）
    chess_prepare_for_pipelining
    chess_loop_range(1,)
    {
        aie::vector<bfloat16, 32> front = aie::load_v<32>(p + i);
        aie::vector<bfloat16, 32> back  = aie::load_v<32>(p + 256 + i);

        aie::store_v(p + i, back);
        aie::store_v(p + 256 + i, front);
    }

}

inline __attribute__((always_inline))
void fft64_cfloat(cbfloat16 * restrict x, cbfloat16 * restrict y)
{
    //----------------------------------------------------------------
    // Stage-0 : Radix-4, Vec 16, 本地 64 点
    //----------------------------------------------------------------
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw0_0[1]	= TWID640_0;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw0_1[1]	= TWID640_1;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw0_2[1]	= TWID640_2;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw1_0[4]	= TWID641_0;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw1_1[4]	= TWID641_1;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw1_2[4]	= TWID641_2;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw2_0[16]	= TWID642_0;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw2_1[16]	= TWID642_1;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw2_2[16]	= TWID642_2;
    
    constexpr unsigned shift_tw = 0;
    constexpr unsigned shift    = 0;
    constexpr bool inv      = false;
    aie::fft_dit_r4_stage<16>(x, tw0_1, tw0_0, tw0_2, 64, shift_tw, shift, inv, y);
    aie::fft_dit_r4_stage<4>(y, tw1_1, tw1_0, tw1_2, 64, shift_tw, shift, inv, x);
    aie::fft_dit_r4_stage<1>(x, tw2_1, tw2_0, tw2_2, 64, shift_tw, shift, inv, y);
    
    //----------------------------------------------------------------
    // FFTSHIFT: swap first 128 and last 128 complex samples
    //----------------------------------------------------------------
    bfloat16* p = reinterpret_cast<bfloat16*>(y);  // 512 bfloat16s = 256 cbfloat16

    for (int i = 0; i < 64 /*= 32 complex × 2*/; i += 32)   // 每次交换 16 复数（32 bfloat16）
        chess_prepare_for_pipelining
        chess_loop_range(1,)
        {
            aie::vector<bfloat16, 32> front = aie::load_v<32>(p + i);        // 0…31 或 32…63
            aie::vector<bfloat16, 32> back  = aie::load_v<32>(p + 64 + i);   // 64…95 或 96…127
            aie::store_v(p + i        , back );
            aie::store_v(p + 64 + i   , front);
        }

}


inline __attribute__((always_inline))
void fft32_cfloat(cbfloat16 * restrict x,cbfloat16 * restrict y)
{
    //----------------------------------------------------------------
    // Stage-0 : Radix-4, Vec 16, 本地 64 点
    //----------------------------------------------------------------
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw0_0[1]	= TWID320_0;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw1_0[2]	= TWID321_0;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw1_1[2]	= TWID321_1;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw1_2[2]	= TWID321_2;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw2_0[8]	= TWID322_0;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw2_1[8]	= TWID322_1;
    alignas(aie::vector_decl_align) static constexpr cbfloat16	tw2_2[8]	= TWID322_2;

    constexpr unsigned shift_tw = 0;
    constexpr unsigned shift    = 0;
    constexpr bool inv      = false;
    aie::fft_dit_r2_stage<16>(x, tw0_0,             32, shift_tw, shift, inv, y);
    aie::fft_dit_r4_stage<4>(y, tw1_1, tw1_0, tw1_2, 32, shift_tw, shift, inv, x);
    aie::fft_dit_r4_stage<1>(x, tw2_1, tw2_0, tw2_2, 32, shift_tw, shift, inv, y);
    
}





#endif // __FAM_FUNCS_H__










