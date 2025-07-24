#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer2_0_down.h"
#include "fam_funcs.h"
#include "parameters.h"

/*──────────────────────── constructor ───────────────────────*/
layer2_0_down::layer2_0_down(int xoff) : m_xoff(xoff)
{
  aie::set_rounding   (aie::rounding_mode::symmetric_inf);
  aie::set_saturation (aie::saturation_mode::saturate);
}


//24*32*32=24576
//16*16*16=4096
void layer2_0_down::run(
        adf::input_buffer<bfloat16, adf::extents<384>>& wbuf,
        input_stream<bfloat16>*                         sin ,
        adf::output_buffer<bfloat16, adf::extents<4096>>& buf)
{
  /* ──────────────── 型别速记 ──────────────── */
  using v32   = aie::vector<bfloat16,32>;
  using v16   = aie::vector<bfloat16,16>;
  using v8    = aie::vector<bfloat16, 8>;
  using acc16 = aie::accum <accfloat ,16>;

  const v16 Z16 = aie::zeros<bfloat16,16>();
  auto it = aie::begin_restrict_vector<8>(buf);

  /* ──────────────── 单行环形缓冲 ────────────
   * 为了保持与前面示例一致仍用 3 行，但 1×1 核只用 rows[1]
   */
  alignas(32) v8 row0[32], row1[32], row2[32];
  v8* rows[3] = {row0,row1,row2};

  /* 每行读取：24 × readincr_v<32> → 768 bf16 */
  auto load_row = [&](v8* dst){
    /* 1×1 且 P=0，不需要左右 pad，直接写满 32 × 24 / 8 = 96 片段
       但保持 24 片段 × v32 ← 8IC × 32pix 读取方式 */
    for(int seg=0; seg<24; ++seg){
      v32 v = readincr_v<32>(sin);        // 32 × bf16
      /* 如需旁路输入行，可 writeincr(sout,v); */
      //writeincr(sout,v);
      dst[seg] = v.extract<8>(0);         // 只演示 8IC，真实场景应展开 24IC
    }
  };

  /* 预热：读第 0,1 行 */
  load_row(row1);
  load_row(row2);

  /* ──────────────── 主循环：16 × 16 ──────────────── */
  for(int r=0; r<16; ++r){
    for(int c=0; c<16; ++c){

      /* 4 组累加器：16 OC = 4 × 4OC-blk */
      acc16 acc0 = aie::zeros<accfloat,16>(),
            acc1 = aie::zeros<accfloat,16>(),
            acc2 = aie::zeros<accfloat,16>(),
            acc3 = aie::zeros<accfloat,16>();

      /* 权重游标复位 */
      auto itap = aie::begin_restrict_vector<16>(wbuf);

      /* 3 × 8IC-blk  (24 IC) */
      for(int ib=0; ib<3; ++ib){
        /* 取输入像素：stride = 2 ⇒ 列索引 = 2*c */
        v8 px0  = rows[1][c<<1];
        v16 pair = aie::concat(px0, px0);         // 8 → 16
        v32 vin;  vin.insert(0,pair); vin.insert(1,pair);

        /* 对应 4 × v16 权重：8IC → 4OC */
        v16 w00 = *(itap++);   // OC0-3
        v16 w01 = *(itap++);   // OC4-7
        v16 w02 = *(itap++);   // OC8-11
        v16 w03 = *(itap++);   // OC12-15

        v32 W0 = aie::concat(w00, Z16);
        v32 W1 = aie::concat(w01, Z16);
        v32 W2 = aie::concat(w02, Z16);
        v32 W3 = aie::concat(w03, Z16);

        /* 单次 MAC：vin × W? */
        acc0 = mac_4x8_8x4(vin, W0, acc0);
        acc1 = mac_4x8_8x4(vin, W1, acc1);
        acc2 = mac_4x8_8x4(vin, W2, acc2);
        acc3 = mac_4x8_8x4(vin, W3, acc3);
      }

      /* ─────── ReLU + 写回 ─────── */
      v8 o0 = acc0.to_vector<bfloat16>().extract<8>(0);   // OC0-3
      v8 o1 = acc1.to_vector<bfloat16>().extract<8>(0);   // OC4-7
      v8 o2 = acc2.to_vector<bfloat16>().extract<8>(0);   // OC8-11
      v8 o3 = acc3.to_vector<bfloat16>().extract<8>(0);   // OC12-15

      v8 out0 = aie::max(aie::shuffle_down_fill(o0, o1, 4), bfloat16(0)); // 0-7
      v8 out1 = aie::max(aie::shuffle_down_fill(o2, o3, 4), bfloat16(0)); // 8-15

      *(it++) = out0;
      *(it++) = out1;
    }

    /* ────────────── 行滚动：stride = 2 ───────────── */
    if(r < 15){
      /* 下移 2 行：连续两次 load_row() */
      rows[0] = rows[1];
      rows[1] = rows[2];
      load_row(rows[2]);          // 第 2r+1 行（丢弃）
      rows[0] = rows[1];
      rows[1] = rows[2];
      load_row(rows[2]);          // 第 2r+2 行（用于下次输出）
    }
  }
}



