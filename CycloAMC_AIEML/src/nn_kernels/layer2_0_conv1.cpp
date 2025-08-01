#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer2_0_conv1.h"
#include "fam_funcs.h"
#include "parameters.h"

/*──────────────────────── constructor ───────────────────────*/
layer2_0_conv1::layer2_0_conv1(int xoff) : m_xoff(xoff)
{
  aie::set_rounding   (aie::rounding_mode::symmetric_inf);
  aie::set_saturation (aie::saturation_mode::saturate);
}


//24*32*32=24576
//16*16*16=4096
void layer2_0_conv1::run(
        adf::input_buffer<bfloat16, adf::extents<3456>>& wbuf,
        input_stream<bfloat16>*                         sin ,
        output_stream<bfloat16>*                        sout)
{
  /* ──────────────── 型别速记 ──────────────── */
  using v32   = aie::vector<bfloat16,32>;
  using v16   = aie::vector<bfloat16,16>;
  using v8    = aie::vector<bfloat16, 8>;
  using acc16 = aie::accum <accfloat ,16>;

  const v16 Z16 = aie::zeros<bfloat16,16>();

  /* ──────────────── 3-行环形缓冲 ──────────── */
  alignas(32) v8 row0[34], row1[34], row2[34];
  v8* rows[3] = {row0,row1,row2};

  /* 读取一整行：24 × readincr_v<32> → 768 bf16 */
  auto load_row = [&](v8* dst){
    dst[0] = Z16.extract<8>(0);           // 左 pad
    for(int seg=0; seg<24; ++seg){
      v32 v = readincr_v<32>(sin);        // 32 × bf16
      /* 可旁路输出原始特征，若不需要可删掉 */
      //writeincr(sout, v);
      dst[1 + seg] = v.extract<8>(0);     // 仅示例保留 IC0-7
    }
    dst[25] = Z16.extract<8>(0);          // 右 pad
  };

  /* 预热：pad 行 + row-0 + row-1 */
  for(int i=0;i<34;++i) row0[i] = Z16.extract<8>(0);
  load_row(row1);     /* input row-0 */
  load_row(row2);     /* input row-1 */

  /* ──────────────── 主循环：16 × 16 ──────────────── */
  for(int r=0; r<16; ++r){
    for(int c=0; c<16; ++c){

      /* 4 组累加器：每组 4-OC → 16-OC 总计 */
      acc16 acc0 = aie::zeros<accfloat,16>(),
            acc1 = aie::zeros<accfloat,16>(),
            acc2 = aie::zeros<accfloat,16>(),
            acc3 = aie::zeros<accfloat,16>();

      /* 权重游标每像素从头开始 */
      auto itap = aie::begin_restrict_vector<16>(wbuf);

      /* 3 × 8IC-block  (24-IC) */
      for(int ib=0; ib<3; ++ib){
        v32 vin[3];
        for(int kr=0; kr<3; ++kr){
          /* stride=2 ⇒ 输入列索引 = 1 + 2*c */
          v8  px0  = rows[kr][1 + (c<<1)];
          v16 pair = aie::concat(px0, px0);    // 8-lane → 16-lane
          vin[kr].insert(0, pair);             // (0..15)
          vin[kr].insert(1, pair);             // (16..31)
        }

        /* kernel-column 0..2 */
        for(int kc = 0; kc < 3; ++kc){
          for(int kr = 0; kr < 3; ++kr){                 // 新增 kr = kernel row
            /* 4 × v16 : 8IC → 4OC (OC0‑15) */
            v16 w00 = *(itap++);   // OC0‑3
            v16 w01 = *(itap++);   // OC4‑7
            v16 w02 = *(itap++);   // OC8‑11
            v16 w03 = *(itap++);   // OC12‑15
        
            /* 复制 16lane→32lane，避免丢半边 */
            v32 W0 = aie::concat(w00, w00);
            v32 W1 = aie::concat(w01, w01);
            v32 W2 = aie::concat(w02, w02);
            v32 W3 = aie::concat(w03, w03);
        
            /* 逐行 MAC —— 现在每行用自己的权重 */
            acc0 = mac_4x8_8x4(vin[kr], W0, acc0);
            acc1 = mac_4x8_8x4(vin[kr], W1, acc1);
            acc2 = mac_4x8_8x4(vin[kr], W2, acc2);
            acc3 = mac_4x8_8x4(vin[kr], W3, acc3);
          }
        }
      }

      /* ─────── ReLU + 写回 (2 × writeincr → 4096 总) ─────── */
      v8 o0 = acc0.to_vector<bfloat16>().extract<8>(0);   // OC0-3
      v8 o1 = acc1.to_vector<bfloat16>().extract<8>(0);   // OC4-7
      v8 o2 = acc2.to_vector<bfloat16>().extract<8>(0);   // OC8-11
      v8 o3 = acc3.to_vector<bfloat16>().extract<8>(0);   // OC12-15

      v8 out0 = aie::max(aie::shuffle_down_fill(o0, o1, 4), bfloat16(0)); // 0-7
      v8 out1 = aie::max(aie::shuffle_down_fill(o2, o3, 4), bfloat16(0)); // 8-15

      writeincr(sout, out0);   // 每像素 2 次写回
      writeincr(sout, out1);
    }

    /* ────────────── 行滚动：stride = 2 ───────────── */
    if(r < 15){
      /* 每输出行下移 2 输入行：重复两次 pointer-rotate + load */
      for(int s=0; s<2; ++s){
        
        rows[0] = rows[1];
        rows[1] = rows[2];
        
        load_row(rows[2]);     // 读下一原始输入行
      }
    }
  }
}