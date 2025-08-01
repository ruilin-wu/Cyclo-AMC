#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer3_0_conv2.h"
#include "fam_funcs.h"
#include "parameters.h"

/*──────────────────────── constructor ───────────────────────*/
layer3_0_conv2::layer3_0_conv2(int xoff) : m_xoff(xoff)
{
  aie::set_rounding   (aie::rounding_mode::symmetric_inf);
  aie::set_saturation (aie::saturation_mode::saturate);
}



void layer3_0_conv2::run(
        adf::input_buffer<bfloat16, adf::extents<15552>>& wbuf,
        input_stream<bfloat16>*                          sin ,
        adf::input_buffer<bfloat16, adf::extents<1536>>& residual,
        output_stream<bfloat16>*                         sout)
{
  /* ──────────────── 型别速记 ──────────────── */
  using v32   = aie::vector<bfloat16,32>;
  using v64   = aie::vector<bfloat16,64>;
  using v16   = aie::vector<bfloat16,16>;
  using v8    = aie::vector<bfloat16, 8>;
  using acc16 = aie::accum <accfloat ,16>;

  const v16 Z16 = aie::zeros<bfloat16,16>();
  const v32 Z32 = aie::zeros<bfloat16,32>();
  const v64 Z64 = aie::zeros<bfloat16,64>();
  auto rptr = aie::begin_restrict_vector<8>(residual);   // 8192 / 8 = 1024 个 v8

  /* ───────────── 3-行环形缓冲 (8+2=10) ───────────── */
  alignas(32) v8 row0[10], row1[10], row2[10];
  v8* rows[3] = {row0,row1,row2};

  /* 每行读取：72IC × 8W / 32 = 18 次 readincr_v<32> */
  auto load_row = [&](v8* dst){
    dst[0] = Z16.extract<8>(0);               // 左 pad
    for(int seg=0; seg<18; ++seg){
      v32 v = readincr_v<32>(sin);
      
      //writeincr(sout, v);
      dst[1 + seg] = v.extract<8>(0);         // 仅示例保留 IC0-7
    }
    dst[19] = Z16.extract<8>(0);              // 右 pad
  };

  /* 预热：pad 行 + 第 0、1 行 */
  for(int i=0;i<10;++i) row0[i] = Z16.extract<8>(0);
  load_row(row1);
  load_row(row2);

  /* ──────────────── 主循环：8 × 8 ──────────────── */
  for(int r=0; r<8; ++r){
    for(int c=0; c<8; ++c){

      /* 6 组累加器：24OC = 6 × 4OC */
      acc16 acc0 = aie::zeros<accfloat,16>(),
            acc1 = aie::zeros<accfloat,16>(),
            acc2 = aie::zeros<accfloat,16>(),
            acc3 = aie::zeros<accfloat,16>(),
            acc4 = aie::zeros<accfloat,16>(),
            acc5 = aie::zeros<accfloat,16>();

      /* 权重游标复位 */
      auto itap = aie::begin_restrict_vector<32>(wbuf);

      /* 9 × 8IC-block  (72 IC) */
      for(int ib=0; ib<9; ++ib){
        v32 vin[3];
        for(int kr=0; kr<3; ++kr){
          /* stride = 1 ⇒ 输入列索引 = 1 + c */
          v8 px  = rows[kr][1 + c];
          v16 pair = aie::concat(px, px);       // 8 → 16
          vin[kr].insert(0, pair);
          vin[kr].insert(1, pair);
        }

        /* kernel-column 0..2 */
        for(int kc = 0; kc < 3; ++kc){
          for(int krw = 0; krw < 3; ++krw){          // 新增 kernel‑row

            /* 每行 24OC：6 × v32，完整 32‑lane 不丢弃 */
            v32 W0 = *(itap++);   // OC0‑3
            v32 W1 = *(itap++);   // OC4‑7
            v32 W2 = *(itap++);   // OC8‑11
            v32 W3 = *(itap++);   // OC12‑15
            v32 W4 = *(itap++);   // OC16‑19
            v32 W5 = *(itap++);   // OC20‑23

            acc0 = mac_4x8_8x4(vin[krw], W0, acc0);
            acc1 = mac_4x8_8x4(vin[krw], W1, acc1);
            acc2 = mac_4x8_8x4(vin[krw], W2, acc2);
            acc3 = mac_4x8_8x4(vin[krw], W3, acc3);
            acc4 = mac_4x8_8x4(vin[krw], W4, acc4);
            acc5 = mac_4x8_8x4(vin[krw], W5, acc5);
          }
        }
      }

      /* ─────── ReLU + 写回 (3× writeincr → 1 536 总) ─────── */
      v8 o0 = acc0.to_vector<bfloat16>().extract<8>(0);   // OC0-3
      v8 o1 = acc1.to_vector<bfloat16>().extract<8>(0);   // OC4-7
      v8 o2 = acc2.to_vector<bfloat16>().extract<8>(0);   // OC8-11
      v8 o3 = acc3.to_vector<bfloat16>().extract<8>(0);   // OC12-15
      v8 o4 = acc4.to_vector<bfloat16>().extract<8>(0);   // OC16-19
      v8 o5 = acc5.to_vector<bfloat16>().extract<8>(0);   // OC20-23

      v8 out0 = aie::shuffle_down_fill(o0, o1, 4); // 0-7
      v8 out1 = aie::shuffle_down_fill(o2, o3, 4); // 8-15
      v8 out2 = aie::shuffle_down_fill(o4, o5, 4); // 16-23

      
      v8 relu0 = aie::max(aie::add(out0, *(rptr++)), bfloat16(0));
      writeincr(sout, relu0);              // 写出最终结果

      v8 relu1 = aie::max(aie::add(out1, *(rptr++)), bfloat16(0));
      writeincr(sout, relu1);              // 写出最终结果

      v8 relu2 = aie::max(aie::add(out2, *(rptr++)), bfloat16(0));
      writeincr(sout, relu2);              // 写出最终结果
    }

    /* ──────────── 行滚动：stride = 1 ──────────── */
    if(r < 6){
      rows[0] = rows[1];
      rows[1] = rows[2];
      load_row(rows[2]);                // 读下一输入行
    }
  }
}