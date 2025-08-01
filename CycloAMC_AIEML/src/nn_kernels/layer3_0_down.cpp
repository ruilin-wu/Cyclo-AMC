#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer3_0_down.h"
#include "fam_funcs.h"
#include "parameters.h"

/*──────────────────────── constructor ───────────────────────*/
layer3_0_down::layer3_0_down(int xoff) : m_xoff(xoff)
{
  aie::set_rounding   (aie::rounding_mode::symmetric_inf);
  aie::set_saturation (aie::saturation_mode::saturate);
}



void layer3_0_down::run(
        adf::input_buffer<bfloat16, adf::extents<1152>>& wbuf,
        input_stream<bfloat16>*                         sin ,
        adf::output_buffer<bfloat16, adf::extents<1536>>& buf)
{
  /* ──────────────── 型别速记 ──────────────── */
  using v32   = aie::vector<bfloat16,32>;
  using v16   = aie::vector<bfloat16,16>;
  using v8    = aie::vector<bfloat16, 8>;
  using acc16 = aie::accum <accfloat ,16>;

  const v16 Z16 = aie::zeros<bfloat16,16>();
  auto it = aie::begin_restrict_vector<8>(buf);

  /* ────────── 行缓冲：仅需两行 + 占位 ────────── */
  alignas(32) v8 row0[26], row1[26], row2[26];   // 16 列 + 裤兜预留
  v8* rows[3] = {row0,row1,row2};

  /* 每行读取：48IC × 16W / 32 = 24 次 readincr_v<32> */
  auto load_row = [&](v8* dst){
    for(int seg=0; seg<24; ++seg){
      v32 v = readincr_v<32>(sin);          // 32 × bf16
      //writeincr(sout, v);
      dst[seg] = v.extract<8>(0);           // 仅示例：IC0-7
    }
  };

  /* 预热：先读前两行 */
  load_row(row1);     /* row index 0 */
  load_row(row2);     /* row index 1 */

  /* ──────────────── 主循环：8 × 8 ──────────────── */
  for(int r=0; r<8; ++r){
    for(int c=0; c<8; ++c){

      /* 6 组累加器：24 OC = 6 × 4OC-blk */
      acc16 acc0 = aie::zeros<accfloat,16>(),
            acc1 = aie::zeros<accfloat,16>(),
            acc2 = aie::zeros<accfloat,16>(),
            acc3 = aie::zeros<accfloat,16>(),
            acc4 = aie::zeros<accfloat,16>(),
            acc5 = aie::zeros<accfloat,16>();

      /* 权重游标复位 */
      auto itap = aie::begin_restrict_vector<32>(wbuf);

      /* 6 × 8IC-block  (48 IC) */
      for(int ib=0; ib<6; ++ib){
        /* ── 1×1 取单像素 ── */
        v8  px  = rows[1][c<<1];           // stride = 2
        v16 pair = aie::concat(px, px);    // 8 → 16
        v32 vin;  vin.insert(0, pair); vin.insert(1, pair);

        /* 6 × v32 权重：8IC → 4OC × 6 = 24OC */
        v32 W0 = *(itap++);   // OC0‑3
        v32 W1 = *(itap++);   // OC4‑7
        v32 W2 = *(itap++);   // OC8‑11
        v32 W3 = *(itap++);   // OC12‑15
        v32 W4 = *(itap++);   // OC16‑19
        v32 W5 = *(itap++);   // OC20‑23

        /* 单次 MAC（1×1 卷积，无 kr/kc 维度） */
        acc0 = mac_4x8_8x4(vin, W0, acc0);
        acc1 = mac_4x8_8x4(vin, W1, acc1);
        acc2 = mac_4x8_8x4(vin, W2, acc2);
        acc3 = mac_4x8_8x4(vin, W3, acc3);
        acc4 = mac_4x8_8x4(vin, W4, acc4);
        acc5 = mac_4x8_8x4(vin, W5, acc5);
      }

      /* ─────── ReLU + 写回 ─────── */
      v8 o0 = acc0.to_vector<bfloat16>().extract<8>(0);   // OC0-3
      v8 o1 = acc1.to_vector<bfloat16>().extract<8>(0);   // OC4-7
      v8 o2 = acc2.to_vector<bfloat16>().extract<8>(0);   // OC8-11
      v8 o3 = acc3.to_vector<bfloat16>().extract<8>(0);   // OC12-15
      v8 o4 = acc4.to_vector<bfloat16>().extract<8>(0);   // OC16-19
      v8 o5 = acc5.to_vector<bfloat16>().extract<8>(0);   // OC20-23

      v8 out0 = aie::max(aie::shuffle_down_fill(o0, o1, 4), bfloat16(0)); // 0-7
      v8 out1 = aie::max(aie::shuffle_down_fill(o2, o3, 4), bfloat16(0)); // 8-15
      v8 out2 = aie::max(aie::shuffle_down_fill(o4, o5, 4), bfloat16(0)); // 16-23

      *(it++) = out0;
      *(it++) = out1;
      *(it++) = out2;
    }

    /* ──────────── 行滚动：stride = 2 ──────────── */
    if(r < 7){
      /* 下移 2 输入行 */
      rows[0] = rows[1];
      rows[1] = rows[2];
      load_row(rows[2]);              // 丢弃行 (2r+1)

      rows[0] = rows[1];
      rows[1] = rows[2];
      load_row(rows[2]);              // 行 (2r+2) → 下一次用
    }
  }
}