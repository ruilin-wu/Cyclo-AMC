#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer4_0_conv1.h"
#include "fam_funcs.h"
#include "parameters.h"

layer4_0_conv1::layer4_0_conv1(int xoff) : m_xoff(xoff)
{
  aie::set_rounding(aie::rounding_mode::symmetric_inf);
  aie::set_saturation(aie::saturation_mode::saturate);
}

void layer4_0_conv1::run(
  adf::input_buffer<bfloat16, adf::extents<10368>>& wbuf,
  adf::input_buffer<bfloat16, adf::extents<10368>>& wbuf1,
  input_stream<bfloat16>* sin,
  output_stream<bfloat16>* sout)
{
  using v32 = aie::vector<bfloat16,32>;
  using v16 = aie::vector<bfloat16,16>;
  using v8  = aie::vector<bfloat16, 8>;
  using acc16 = aie::accum<accfloat,16>;

  constexpr int COLS = 8;
  constexpr int COLS_PADDED = COLS + 2;
  constexpr int BLKS = 9;

  const v16 Z16 = aie::zeros<bfloat16,16>();

  alignas(32) v8 row0[COLS_PADDED * BLKS],
                row1[COLS_PADDED * BLKS],
                row2[COLS_PADDED * BLKS];
  v8* rows[3] = {row0, row1, row2};

  auto load_row = [&](v8* dst)
  {
    for(int ib = 0; ib < BLKS; ++ib)
      dst[ib] = Z16.extract<8>(0);

    for(int col = 0; col < COLS; ++col)
    {
      int base = (col + 1) * BLKS;
      for(int ib = 0; ib < BLKS; ++ib)
      {
        v8 v = readincr_v<8>(sin);
        dst[base + ib] = v;
      }
    }

    for(int ib = 0; ib < BLKS; ++ib)
      dst[(COLS_PADDED - 1) * BLKS + ib] = Z16.extract<8>(0);
  };

  load_row(row1);
  load_row(row2);

  for(int r = 0; r < 4; ++r)
  {
    for(int c = 0; c < 4; ++c)
    {
      acc16 acc0 = aie::zeros<accfloat,16>(),
            acc1 = aie::zeros<accfloat,16>(),
            acc2 = aie::zeros<accfloat,16>(),
            acc3 = aie::zeros<accfloat,16>(),
            acc4 = aie::zeros<accfloat,16>(),
            acc5 = aie::zeros<accfloat,16>(),
            acc6 = aie::zeros<accfloat,16>(),
            acc7 = aie::zeros<accfloat,16>();

      auto itap0 = aie::begin_restrict_vector<32>(wbuf);
      auto itap1 = aie::begin_restrict_vector<32>(wbuf1);

      for(int ib = 0; ib < BLKS; ++ib)
      {
        v32 vin[3];
        for(int kr = 0; kr < 3; ++kr)
        {
          v8 px = rows[kr][(1 + (c << 1)) * BLKS + ib];
          v16 pair = aie::concat(px, px);
          vin[kr].insert(0, pair);
          vin[kr].insert(1, pair);
        }

        for (int kc = 0; kc < 3; ++kc) {
          for (int krw = 0; krw < 3; ++krw) {
            v32 W0 = *(itap0++);
            v32 W1 = *(itap0++);
            v32 W2 = *(itap0++);
            v32 W3 = *(itap0++);
            v32 W4 = *(itap1++);
            v32 W5 = *(itap1++);
            v32 W6 = *(itap1++);
            v32 W7 = *(itap1++);

            acc0 = mac_4x8_8x4(vin[krw], W0, acc0);
            acc1 = mac_4x8_8x4(vin[krw], W1, acc1);
            acc2 = mac_4x8_8x4(vin[krw], W2, acc2);
            acc3 = mac_4x8_8x4(vin[krw], W3, acc3);
            acc4 = mac_4x8_8x4(vin[krw], W4, acc4);
            acc5 = mac_4x8_8x4(vin[krw], W5, acc5);
            acc6 = mac_4x8_8x4(vin[krw], W6, acc6);
            acc7 = mac_4x8_8x4(vin[krw], W7, acc7);
          }
        }
      }

      v8 lo0 = acc0.to_vector<bfloat16>().extract<8>(0);
      v8 hi0 = acc1.to_vector<bfloat16>().extract<8>(0);
      v8 lo1 = acc2.to_vector<bfloat16>().extract<8>(0);
      v8 hi1 = acc3.to_vector<bfloat16>().extract<8>(0);
      v8 lo2 = acc4.to_vector<bfloat16>().extract<8>(0);
      v8 hi2 = acc5.to_vector<bfloat16>().extract<8>(0);
      v8 lo3 = acc6.to_vector<bfloat16>().extract<8>(0);
      v8 hi3 = acc7.to_vector<bfloat16>().extract<8>(0);

      v8 out0 = aie::max(aie::shuffle_down_fill(lo0, hi0, 4), bfloat16(0));
      v8 out1 = aie::max(aie::shuffle_down_fill(lo1, hi1, 4), bfloat16(0));
      v8 out2 = aie::max(aie::shuffle_down_fill(lo2, hi2, 4), bfloat16(0));
      v8 out3 = aie::max(aie::shuffle_down_fill(lo3, hi3, 4), bfloat16(0));

      writeincr(sout, out0);
      writeincr(sout, out1);
      writeincr(sout, out2);
      writeincr(sout, out3);
    }

    if(r < 3)
    {
      rows[0] = rows[1];
      rows[1] = rows[2];
      load_row(rows[2]);

      rows[0] = rows[1];
      rows[1] = rows[2];
      load_row(rows[2]);
    }
  }
}
