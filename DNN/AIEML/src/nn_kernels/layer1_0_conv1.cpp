#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer1_0_conv1.h"
#include "fam_funcs.h"
#include "parameters.h"

layer1_0_conv1::layer1_0_conv1(int xoff) : m_xoff(xoff)
{
  aie::set_rounding(aie::rounding_mode::symmetric_inf);
  aie::set_saturation(aie::saturation_mode::saturate);
}

void layer1_0_conv1::run(adf::input_buffer<bfloat16, adf::extents<1728>>& wbuf,
                         input_stream<bfloat16>* sin,
                         output_stream<bfloat16>* sout)
{
  using v32 = aie::vector<bfloat16,32>;
  using v64 = aie::vector<bfloat16,64>;
  using v16 = aie::vector<bfloat16,16>;
  using v8  = aie::vector<bfloat16, 8>;
  using acc16 = aie::accum<accfloat,16>;

  const v16 Z16 = aie::zeros<bfloat16,16>();
  const v32 Z32 = aie::zeros<bfloat16,32>();
  const v64 Z64 = aie::zeros<bfloat16,64>();

  alignas(32) v32 row0[34], row1[34], row2[34];
  v32* rows[3] = {row0,row1,row2};

  auto load_row = [&](v32* dst){
    dst[0] = Z64.extract<32>(0);
    for(int seg=0; seg<24; ++seg){
      v32 v = readincr_v<32>(sin);
      dst[1 + seg] = v;
    }
    dst[25] = Z64.extract<32>(0);
  };

  for(int i=0;i<34;++i) row0[i]=Z64.extract<32>(0);
  load_row(row1);
  load_row(row2);

  for(int r=0;r<32;++r){
    for(int c=0;c<32;++c){
      acc16 acc0=aie::zeros<accfloat,16>(),
            acc1=aie::zeros<accfloat,16>();

      auto itap = aie::begin_restrict_vector<16>(wbuf);

      for (int ib = 0; ib < 3; ++ib) {
        v32 vin[3];
        for (int kr = 0; kr < 3; ++kr) {
          v32 col32 = rows[kr][1 + c];
          v8  px8   = col32.extract<8>(ib);
          v16 pair  = aie::concat(px8, px8);
          vin[kr].insert(0, pair);
          vin[kr].insert(1, pair);
        }

        for (int kc = 0; kc < 3; ++kc) {
          for (int krw = 0; krw < 3; ++krw) {
            v16 lo = *(itap++);
            v16 hi = *(itap++);
            v32 w0 = aie::concat(lo, Z32.extract<16>(0));
            v32 w1 = aie::concat(hi, Z32.extract<16>(0));
            acc0 = mac_4x8_8x4(vin[krw], w0, acc0);
            acc1 = mac_4x8_8x4(vin[krw], w1, acc1);
          }
        }
      }

      v8 low  = acc0.to_vector<bfloat16>().extract<8>(0);
      v8 high = acc1.to_vector<bfloat16>().extract<8>(0);
      v8 out  = aie::max(aie::shuffle_down_fill(low, high, 4), bfloat16(0));
      writeincr(sout, out);
    }

    if (r < 30) {
      rows[0] = rows[1];
      rows[1] = rows[2];
      load_row(rows[2]);
    }
  }
}
