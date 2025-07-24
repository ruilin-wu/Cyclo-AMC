#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer4_0_conv1.h"
#include "fam_funcs.h"
#include "parameters.h"

/*──────────────────────── constructor ───────────────────────*/
layer4_0_conv1::layer4_0_conv1(int xoff) : m_xoff(xoff)
{
  aie::set_rounding   (aie::rounding_mode::symmetric_inf);
  aie::set_saturation (aie::saturation_mode::saturate);
}


//24*32*32=24576
//16*16*16=4096
void layer4_0_conv1::run(
        adf::input_buffer<bfloat16, adf::extents<10368>>& wbuf,
        input_stream<bfloat16>*                          sin ,
        output_stream<bfloat16>*                         sout)
{
  /* ───────── 型别速记 ───────── */
  using v32   = aie::vector<bfloat16,32>;
  using v16   = aie::vector<bfloat16,16>;
  using v8    = aie::vector<bfloat16, 8>;
  using acc16 = aie::accum <accfloat ,16>;

  constexpr int COLS        = 8;            // 输入宽
  constexpr int COLS_PADDED = COLS + 2;     // pad 左+右
  constexpr int BLKS        = 9;            // 72 IC / 8

  const v16 Z16 = aie::zeros<bfloat16,16>();

  /* ────── 行缓冲：3 行 × 10 列 × 9 blk ────── */
  alignas(32) v8 row0[COLS_PADDED * BLKS],
                row1[COLS_PADDED * BLKS],
                row2[COLS_PADDED * BLKS];
  v8* rows[3] = {row0,row1,row2};

  /* ────── 读取一行：72 × 8 v8 / 行 ────── */
  auto load_row = [&](v8* dst)
  {
    /* 左 pad：9 个 blk 写 0 */
    for(int ib = 0; ib < BLKS; ++ib)
      dst[ib] = Z16.extract<8>(0);

    /* 有效 8 列 */
    for(int col = 0; col < COLS; ++col)
    {
      int base = (col + 1) * BLKS;          // +1 跳过左 pad
      for(int ib = 0; ib < BLKS; ++ib)
        {
          v8 v = readincr_v<8>(sin);/* 每次读 8 通道 */
          //writeincr(sout, v);
          dst[base + ib] = v;
        }
    }

    /* 右 pad：9 个 blk 写 0 */
    for(int ib = 0; ib < BLKS; ++ib)
      dst[(COLS_PADDED - 1) * BLKS + ib] = Z16.extract<8>(0);
  };

  /* ────── 预热两行 ────── */
  load_row(row1);    // 输入第 0 行
  load_row(row2);    // 输入第 1 行

  /* ────── 主卷积循环：4 × 4 ────── */
  for(int r = 0; r < 4; ++r)
  {
    for(int c = 0; c < 4; ++c)
    {
      /* 4 × 4OC 累加器 → 16OC */
      acc16 acc0 = aie::zeros<accfloat,16>(),
            acc1 = aie::zeros<accfloat,16>(),
            acc2 = aie::zeros<accfloat,16>(),
            acc3 = aie::zeros<accfloat,16>(),
            acc4 = aie::zeros<accfloat,16>(),
            acc5 = aie::zeros<accfloat,16>(),
            acc6 = aie::zeros<accfloat,16>(),
            acc7 = aie::zeros<accfloat,16>();

      auto itap = aie::begin_restrict_vector<16>(wbuf);   // 权重指针

      /* 9 × 8IC block */
      for(int ib = 0; ib < BLKS; ++ib)
      {
        /* 3 行像素 → vin[] */
        v32 vin[3];
        for(int kr = 0; kr < 3; ++kr)
        {
          v8  px   = rows[kr][ (1 + (c << 1)) * BLKS + ib ]; // stride = 2
          v16 pair = aie::concat(px, px);     // 8 → 16 lane
          vin[kr].insert(0, pair);
          vin[kr].insert(1, pair);
        }

        /* kernel col 0..2 */
        for(int kc = 0; kc < 3; ++kc)
        {
          v16 w0 = *(itap++);   // OC0-3
          v16 w1 = *(itap++);   // OC4-7
          v16 w2 = *(itap++);   // OC8-11
          v16 w3 = *(itap++);   // OC12-15
          v16 w4 = *(itap++);   // OC16-19
          v16 w5 = *(itap++);   // OC20-23
          v16 w6 = *(itap++);   // OC24-27
          v16 w7 = *(itap++);   // OC28-31

          v32 W0 = aie::concat(w0, Z16);
          v32 W1 = aie::concat(w1, Z16);
          v32 W2 = aie::concat(w2, Z16);
          v32 W3 = aie::concat(w3, Z16);
          v32 W4 = aie::concat(w4, Z16);
          v32 W5 = aie::concat(w5, Z16);
          v32 W6 = aie::concat(w6, Z16);
          v32 W7 = aie::concat(w7, Z16);

          for(int kr = 0; kr < 3; ++kr)
          {
            acc0 = mac_4x8_8x4(vin[kr], W0, acc0);
            acc1 = mac_4x8_8x4(vin[kr], W1, acc1);
            acc2 = mac_4x8_8x4(vin[kr], W2, acc2);
            acc3 = mac_4x8_8x4(vin[kr], W3, acc3);
            acc4 = mac_4x8_8x4(vin[kr], W4, acc4);
            acc5 = mac_4x8_8x4(vin[kr], W5, acc5);
            acc6 = mac_4x8_8x4(vin[kr], W6, acc6);
            acc7 = mac_4x8_8x4(vin[kr], W7, acc7);
          }
        }
      }

      /* ────── ReLU + 写回 ────── */
      v8 lo0 = acc0.to_vector<bfloat16>().extract<8>(0);  // OC0-3
      v8 hi0 = acc1.to_vector<bfloat16>().extract<8>(0);  // OC4-7
      v8 lo1 = acc2.to_vector<bfloat16>().extract<8>(0);  // OC8-11
      v8 hi1 = acc3.to_vector<bfloat16>().extract<8>(0);  // OC12-15
      v8 lo2 = acc4.to_vector<bfloat16>().extract<8>(0);  // OC16-19
      v8 hi2 = acc5.to_vector<bfloat16>().extract<8>(0);  // OC20-23
      v8 lo3 = acc6.to_vector<bfloat16>().extract<8>(0);  // OC24-27
      v8 hi3 = acc7.to_vector<bfloat16>().extract<8>(0);  // OC28-31
      v8 out0 = aie::max(aie::shuffle_down_fill(lo0, hi0, 4), bfloat16(0));
      v8 out1 = aie::max(aie::shuffle_down_fill(lo1, hi1, 4), bfloat16(0));
      v8 out2 = aie::max(aie::shuffle_down_fill(lo2, hi2, 4), bfloat16(0));
      v8 out3 = aie::max(aie::shuffle_down_fill(lo3, hi3, 4), bfloat16(0));

      writeincr(sout, out0);
      writeincr(sout, out1); 
      writeincr(sout, out2);
      writeincr(sout, out3);
    }

    /* ────── 行滚动 (stride = 2) ────── */
    if(r < 3)
    {
      /* 向下移动两行 */
      rows[0] = rows[1];
      rows[1] = rows[2];
      load_row(rows[2]);     // 丢弃行 2r+1

      rows[0] = rows[1];
      rows[1] = rows[2];
      load_row(rows[2]);     // 读入行 2r+2
    }
  }
}