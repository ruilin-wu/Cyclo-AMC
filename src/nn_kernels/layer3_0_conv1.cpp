#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer3_0_conv1.h"
#include "fam_funcs.h"
#include "parameters.h"

/*──────────────────────── constructor ───────────────────────*/
layer3_0_conv1::layer3_0_conv1(int xoff) : m_xoff(xoff)
{
  aie::set_rounding   (aie::rounding_mode::symmetric_inf);
  aie::set_saturation (aie::saturation_mode::saturate);
}


void layer3_0_conv1::run(
        adf::input_buffer<bfloat16, adf::extents<10368>>& wbuf,
        input_stream<bfloat16>*                          sin ,
        output_stream<bfloat16>*                         sout)
{
  /* ──────────────── 型别速记 ──────────────── */
  using v32   = aie::vector<bfloat16,32>;
  using v16   = aie::vector<bfloat16,16>;
  using v8    = aie::vector<bfloat16, 8>;
  using acc16 = aie::accum <accfloat ,16>;

  const v16 Z16 = aie::zeros<bfloat16,16>();

  /* ──────────────── 3-行环形缓冲 ──────────── */
  alignas(32) v8 row0[34], row1[34], row2[34];   // 1+16+1 ≤ 34，沿用旧尺寸
  v8* rows[3] = {row0,row1,row2};

  /* 读取一行：48 IC × 16 / 32 = 24 次 readincr_v<32> → 768 bf16 */
  auto load_row = [&](v8* dst){
    dst[0] = Z16.extract<8>(0);                  // 左 pad
    for(int seg=0; seg<24; ++seg){
      v32 v = readincr_v<32>(sin);               // 32 × bf16
      /* 如果想旁路输入，可启用： writeincr(sout, v); */
      //writeincr(sout, v);
      dst[1 + seg] = v.extract<8>(0);            // 这里只示例保留 IC0-7
    }
    dst[25] = Z16.extract<8>(0);                 // 右 pad
  };

  /* 预热：pad 行 + 第 0,1 行 */
  for(int i=0; i<34; ++i) row0[i] = Z16.extract<8>(0);
  load_row(row1);
  load_row(row2);

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
      auto itap = aie::begin_restrict_vector<16>(wbuf);

      /* 6 × 8IC-blk  (48 IC) */
      for(int ib=0; ib<6; ++ib){
        /* vin[kr]：将 8IC 像素复制成 32-lane */
        v32 vin[3];
        for(int kr=0; kr<3; ++kr){
          /* stride = 2 ⇒ 列索引 = 1 + 2c */
          v8  px  = rows[kr][1 + (c<<1)];
          v16 pair = aie::concat(px, px);        // 8 → 16
          vin[kr].insert(0, pair);
          vin[kr].insert(1, pair);
        }

        /* kernel-column 0-2 */
        for(int kc=0; kc<3; ++kc){
          /* 6 × v16：8IC → 4OC */
          v16 w00 = *(itap++);   // OC0-3
          v16 w01 = *(itap++);   // OC4-7
          v16 w02 = *(itap++);   // OC8-11
          v16 w03 = *(itap++);   // OC12-15
          v16 w04 = *(itap++);   // OC16-19
          v16 w05 = *(itap++);   // OC20-23

          v32 W0 = aie::concat(w00, Z16);
          v32 W1 = aie::concat(w01, Z16);
          v32 W2 = aie::concat(w02, Z16);
          v32 W3 = aie::concat(w03, Z16);
          v32 W4 = aie::concat(w04, Z16);
          v32 W5 = aie::concat(w05, Z16);

          /* 3 行 MAC */
          for(int kr=0; kr<3; ++kr){



            acc0 = mac_4x8_8x4(vin[kr], W0, acc0);
            acc1 = mac_4x8_8x4(vin[kr], W1, acc1);
            acc2 = mac_4x8_8x4(vin[kr], W2, acc2);
            acc3 = mac_4x8_8x4(vin[kr], W3, acc3);
            acc4 = mac_4x8_8x4(vin[kr], W4, acc4);
            acc5 = mac_4x8_8x4(vin[kr], W5, acc5);
          }
        }
      }

      /* ─────── ReLU + 写回 (3 × writeincr → 1 536 总) ─────── */
      v8 o0 = acc0.to_vector<bfloat16>().extract<8>(0);   // OC0-3
      v8 o1 = acc1.to_vector<bfloat16>().extract<8>(0);   // OC4-7
      v8 o2 = acc2.to_vector<bfloat16>().extract<8>(0);   // OC8-11
      v8 o3 = acc3.to_vector<bfloat16>().extract<8>(0);   // OC12-15
      v8 o4 = acc4.to_vector<bfloat16>().extract<8>(0);   // OC16-19
      v8 o5 = acc5.to_vector<bfloat16>().extract<8>(0);   // OC20-23

      v8 out0 = aie::max(aie::shuffle_down_fill(o0, o1, 4), bfloat16(0)); // 0-7
      v8 out1 = aie::max(aie::shuffle_down_fill(o2, o3, 4), bfloat16(0)); // 8-15
      v8 out2 = aie::max(aie::shuffle_down_fill(o4, o5, 4), bfloat16(0)); // 16-23

      writeincr(sout, out0);
      writeincr(sout, out1);
      writeincr(sout, out2);
    }

    /* ──────────── 行滚动：stride = 2 ──────────── */
    if(r < 7){
      /* 每输出行下移 2 输入行：重复两次 rotate + load */
      for(int s=0; s<2; ++s){
        rows[0] = rows[1];
        rows[1] = rows[2];
        load_row(rows[2]);
      }
    }
  }
}