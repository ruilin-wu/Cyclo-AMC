#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer2_0_down.h"
#include "fam_funcs.h"
#include "parameters.h"

layer2_0_down::layer2_0_down(int xoff) : m_xoff(xoff)
{
  aie::set_rounding(aie::rounding_mode::symmetric_inf);
  aie::set_saturation(aie::saturation_mode::saturate);
}

void layer2_0_down::run(
  adf::input_buffer<bfloat16, adf::extents<384>>& wbuf,
  input_stream<bfloat16>* sin,
  adf::output_buffer<bfloat16, adf::extents<4096>>& buf)
{
  using v32 = aie::vector<bfloat16,32>;
  using v16 = aie::vector<bfloat16,16>;
  using v8  = aie::vector<bfloat16, 8>;
  using acc16 = aie::accum<accfloat,16>;

  const v16 Z16 = aie::zeros<bfloat16,16>();
  auto it = aie::begin_restrict_vector<8>(buf);

  alignas(32) v8 row0[32], row1[32], row2[32];
  v8* rows[3] = {row0,row1,row2};

  auto load_row = [&](v8* dst){
    for(int seg=0; seg<24; ++seg){
      v32 v = readincr_v<32>(sin);
      dst[seg] = v.extract<8>(0);
    }
  };

  load_row(row1);
  load_row(row2);

  for(int r=0; r<16; ++r){
    for(int c=0; c<16; ++c){
      acc16 acc0 = aie::zeros<accfloat,16>(),
            acc1 = aie::zeros<accfloat,16>(),
            acc2 = aie::zeros<accfloat,16>(),
            acc3 = aie::zeros<accfloat,16>();

      auto itap = aie::begin_restrict_vector<32>(wbuf);

      for(int ib=0; ib<3; ++ib){
        v8 px0  = rows[1][c<<1];
        v16 pair = aie::concat(px0, px0);
        v32 vin; vin.insert(0,pair); vin.insert(1,pair);

        v32 W0 = *(itap++);
        v32 W1 = *(itap++);
        v32 W2 = *(itap++);
        v32 W3 = *(itap++);

        acc0 = mac_4x8_8x4(vin, W0, acc0);
        acc1 = mac_4x8_8x4(vin, W1, acc1);
        acc2 = mac_4x8_8x4(vin, W2, acc2);
        acc3 = mac_4x8_8x4(vin, W3, acc3);
      }

      v8 o0 = acc0.to_vector<bfloat16>().extract<8>(0);
      v8 o1 = acc1.to_vector<bfloat16>().extract<8>(0);
      v8 o2 = acc2.to_vector<bfloat16>().extract<8>(0);
      v8 o3 = acc3.to_vector<bfloat16>().extract<8>(0);

      v8 out0 = aie::max(aie::shuffle_down_fill(o0, o1, 4), bfloat16(0));
      v8 out1 = aie::max(aie::shuffle_down_fill(o2, o3, 4), bfloat16(0));

      *(it++) = out0;
      *(it++) = out1;
    }

    if(r < 15){
      rows[0] = rows[1];
      rows[1] = rows[2];
      load_row(rows[2]);
      rows[0] = rows[1];
      rows[1] = rows[2];
      load_row(rows[2]);
    }
  }
}
