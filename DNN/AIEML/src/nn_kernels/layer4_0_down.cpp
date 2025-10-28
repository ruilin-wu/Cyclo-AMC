#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer4_0_down.h"
#include "fam_funcs.h"
#include "parameters.h"

layer4_0_down::layer4_0_down(int xoff) : m_xoff(xoff)
{
  aie::set_rounding(aie::rounding_mode::symmetric_inf);
  aie::set_saturation(aie::saturation_mode::saturate);
}

void layer4_0_down::run(
  adf::input_buffer<bfloat16, adf::extents<2304>>& wbuf,
  input_stream<bfloat16>* sin,
  adf::output_buffer<bfloat16, adf::extents<512>>& buf)
{
  using v32 = aie::vector<bfloat16,32>;
  using v16 = aie::vector<bfloat16,16>;
  using v8  = aie::vector<bfloat16, 8>;
  using acc16 = aie::accum<accfloat,16>;

  constexpr int IN_W = 8;
  constexpr int OUT_W = 4;
  constexpr int BLKS = 9;
  const v16 Z16 = aie::zeros<bfloat16,16>();

  auto it = aie::begin_restrict_vector<8>(buf);

  alignas(32) v8 row0[IN_W * BLKS], row1[IN_W * BLKS];
  v8* rows[2] = {row0, row1};

  auto load_row = [&](v8* dst){
    for(int col = 0; col < IN_W; ++col){
      int base = col * BLKS;
      for(int ib = 0; ib < BLKS; ++ib){
        v8 v = readincr_v<8>(sin);
        dst[base + ib] = v;
      }
    }
  };

  load_row(row0);
  load_row(row1);

  for(int r = 0; r < 4; ++r){
    for(int c = 0; c < 4; ++c){
      acc16 acc0 = aie::zeros<accfloat,16>(),
            acc1 = aie::zeros<accfloat,16>(),
            acc2 = aie::zeros<accfloat,16>(),
            acc3 = aie::zeros<accfloat,16>(),
            acc4 = aie::zeros<accfloat,16>(),
            acc5 = aie::zeros<accfloat,16>(),
            acc6 = aie::zeros<accfloat,16>(),
            acc7 = aie::zeros<accfloat,16>();

      auto itap = aie::begin_restrict_vector<32>(wbuf);

      for(int ib = 0; ib < BLKS; ++ib){
        v8 px = rows[0][(c << 1) * BLKS + ib];
        v16 pair = aie::concat(px, px);
        v32 vin; vin.insert(0, pair); vin.insert(1, pair);

        v32 W0 = *(itap++);
        v32 W1 = *(itap++);
        v32 W2 = *(itap++);
        v32 W3 = *(itap++);
        v32 W4 = *(itap++);
        v32 W5 = *(itap++);
        v32 W6 = *(itap++);
        v32 W7 = *(itap++);

        acc0 = mac_4x8_8x4(vin, W0, acc0);
        acc1 = mac_4x8_8x4(vin, W1, acc1);
        acc2 = mac_4x8_8x4(vin, W2, acc2);
        acc3 = mac_4x8_8x4(vin, W3, acc3);
        acc4 = mac_4x8_8x4(vin, W4, acc4);
        acc5 = mac_4x8_8x4(vin, W5, acc5);
        acc6 = mac_4x8_8x4(vin, W6, acc6);
        acc7 = mac_4x8_8x4(vin, W7, acc7);
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

      *(it++) = out0;
      *(it++) = out1;
      *(it++) = out2;
      *(it++) = out3;
    }

    if(r < 3){
      load_row(row0);
      load_row(row1);
    }
  }
}
