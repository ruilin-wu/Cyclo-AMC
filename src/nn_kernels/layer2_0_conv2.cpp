#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer2_0_conv2.h"
#include "fam_funcs.h"
#include "parameters.h"

/*──────────────────────── constructor ───────────────────────*/
layer2_0_conv2::layer2_0_conv2(int xoff) : m_xoff(xoff)
{
  aie::set_rounding   (aie::rounding_mode::symmetric_inf);
  aie::set_saturation (aie::saturation_mode::saturate);
}


//48*16*16=12288
//16*16*16=4096
void layer2_0_conv2::run(
        adf::input_buffer<bfloat16, adf::extents<6912>>& wbuf,
        input_stream<bfloat16>*                         sin ,
        adf::input_buffer<bfloat16, adf::extents<4096>>& residual,
        output_stream<bfloat16>*                        sout)
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

  /* ──────────────── 3-行环形缓冲 ──────────── */
  alignas(32) v8 row0[34], row1[34], row2[34];   // 16(宽) + 2(pad)  ≤ 34
  v8* rows[3] = {row0,row1,row2};

  /* 每行读取： (48 IC × 16) / 32 = 24 次 readincr_v<32> */
  auto load_row = [&](v8* dst){
    dst[0] = Z16.extract<8>(0);                   // 左 pad
    for(int seg=0; seg<24; ++seg){
      v32 v = readincr_v<32>(sin);
      /* 若需旁路输出原特征，可启用下一行 */
      //writeincr(sout, v);
      dst[1 + seg] = v.extract<8>(0);             // 仅示例：IC0-7
    }
    dst[25] = Z16.extract<8>(0);                  // 右 pad
  };

  /* 预热：pad 行 + 第 0,1 行 */
  for(int i=0;i<34;++i) row0[i] = Z16.extract<8>(0);
  load_row(row1);
  load_row(row2);

  /* ──────────────── 主循环：16 × 16 ──────────────── */
  for(int r=0; r<16; ++r){
    for(int c=0; c<16; ++c){

      /* 4 组累加器：16-OC = 4 × 4OC-blk */
      acc16 acc0 = aie::zeros<accfloat,16>(),
            acc1 = aie::zeros<accfloat,16>(),
            acc2 = aie::zeros<accfloat,16>(),
            acc3 = aie::zeros<accfloat,16>();

      /* 权重游标复位 */
      auto itap = aie::begin_restrict_vector<16>(wbuf);

      /* 6 × 8IC-blk  (48-IC) */
      for(int ib=0; ib<6; ++ib){
        v32 vin[3];
        for(int kr=0; kr<3; ++kr){
          v8  px0  = rows[kr][1 + c];             // stride = 1
          v16 pair = aie::concat(px0, px0);
          vin[kr].insert(0, pair);                // (0..15)
          vin[kr].insert(1, pair);                // (16..31)
        }

        /* kernel-column 0..2 */
        for(int kc=0; kc<3; ++kc){
          /* 4 × v16 : 8IC → 4OC */
          v16 w00 = *(itap++);   // OC0-3
          v16 w01 = *(itap++);   // OC4-7
          v16 w02 = *(itap++);   // OC8-11
          v16 w03 = *(itap++);   // OC12-15

          v32 W0 = aie::concat(w00, Z16);
          v32 W1 = aie::concat(w01, Z16);
          v32 W2 = aie::concat(w02, Z16);
          v32 W3 = aie::concat(w03, Z16);

          /* 3 行 MAC */
          for(int kr=0; kr<3; ++kr){
            acc0 = mac_4x8_8x4(vin[kr], W0, acc0);
            acc1 = mac_4x8_8x4(vin[kr], W1, acc1);
            acc2 = mac_4x8_8x4(vin[kr], W2, acc2);
            acc3 = mac_4x8_8x4(vin[kr], W3, acc3);
          }
        }
      }

      /* ─────── ReLU + 写回 ─────── */
      v8 o0 = acc0.to_vector<bfloat16>().extract<8>(0);   // OC0-3
      v8 o1 = acc1.to_vector<bfloat16>().extract<8>(0);   // OC4-7
      v8 o2 = acc2.to_vector<bfloat16>().extract<8>(0);   // OC8-11
      v8 o3 = acc3.to_vector<bfloat16>().extract<8>(0);   // OC12-15

      v8 out0 = aie::shuffle_down_fill(o0, o1, 4); // 0-7
      v8 out1 = aie::shuffle_down_fill(o2, o3, 4); // 8-15

      v8 relu0 = aie::max(aie::add(out0, *(rptr++)), bfloat16(0));
      writeincr(sout, relu0);              // 写出最终结果

      v8 relu1 = aie::max(aie::add(out1, *(rptr++)), bfloat16(0));
      writeincr(sout, relu1);              // 写出最终结果
    }

    /* ─────────── 行滚动：stride = 1 ─────────── */
    if(r < 14){
      rows[0] = rows[1];
      rows[1] = rows[2];
      load_row(rows[2]);     // 读下一行
    }
  }
}