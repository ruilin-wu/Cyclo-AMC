#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer3_0_conv2.h"
#include "fam_funcs.h"
#include "parameters.h"

layer3_0_conv2::layer3_0_conv2(int xoff) : m_xoff(xoff)
{
  aie::set_rounding(aie::rounding_mode::symmetric_inf);
  aie::set_saturation(aie::saturation_mode::saturate);
}

void layer3_0_conv2::run(
  adf::input_buffer<bfloat16, adf::extents<15552>>& wbuf,
  input_stream<bfloat16>* sin,
  adf::input_buffer<bfloat16, adf::extents<1536>>& residual,
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
  auto rptr = aie::begin_restrict_vector<8>(residual);

  alignas(32) v8 row0[10], row1[10], row2[10];
  v8* rows[3] = {row0,row1,row2};

  auto load_row = [&](v8* dst){
    dst[0] = Z16.extract<8>(0);
    for(int seg=0; seg<18; ++seg){
      v32 v = readincr_v<32>(sin);
      dst[1 + seg] = v.extract<8>(0);
    }
    dst[19] = Z16.extract<8>(0);
  };

  for(int i=0; i<10; ++i) row0[i] = Z16.extract<8>(0);
  load_row(row1);
  load_row(row2);

  for(int r=0; r<8; ++r){
    for(int c=0; c<8; ++c){
      acc16 acc0 = aie::zeros<accfloat,16>(),
            acc1 = aie::zeros<accfloat,16>(),
            acc2 = aie::zeros<accfloat,16>(),
            acc3 = aie::zeros<accfloat,16>(),
            acc4 = aie::zeros<accfloat,16>(),
            acc5 = aie::zeros<accfloat,16>();

      auto itap = aie::begin_restrict_vector<32>(wbuf);

      for(int ib=0; ib<9; ++ib){
        v32 vin[3];
        for(int kr=0; kr<3; ++kr){
          v8 px = rows[kr][1 + c];
          v16 pair = aie::concat(px, px);
          vin[kr].insert(0, pair);
          vin[kr].insert(1, pair);
        }

        for(int kc = 0; kc < 3; ++kc){
          for(int krw = 0; krw < 3; ++krw){
            v32 W0 = *(itap++);
            v32 W1 = *(itap++);
            v32 W2 = *(itap++);
            v32 W3 = *(itap++);
            v32 W4 = *(itap++);
            v32 W5 = *(itap++);
            acc0 = mac_4x8_8x4(vin[krw], W0, acc0);
            acc1 = mac_4x8_8x4(vin[krw], W1, acc1);
            acc2 = mac_4x8_8x4(vin[krw], W2, acc2);
            acc3 = mac_4x8_8x4(vin[krw], W3, acc3);
            acc4 = mac_4x8_8x4(vin[krw], W4, acc4);
            acc5 = mac_4x8_8x4(vin[krw], W5, acc5);
          }
        }
      }

      v8 o0 = acc0.to_vector<bfloat16>().extract<8>(0);
      v8 o1 = acc1.to_vector<bfloat16>().extract<8>(0);
      v8 o2 = acc2.to_vector<bfloat16>().extract<8>(0);
      v8 o3 = acc3.to_vector<bfloat16>().extract<8>(0);
      v8 o4 = acc4.to_vector<bfloat16>().extract<8>(0);
      v8 o5 = acc5.to_vector<bfloat16>().extract<8>(0);

      v8 out0 = aie::shuffle_down_fill(o0, o1, 4);
      v8 out1 = aie::shuffle_down_fill(o2, o3, 4);
      v8 out2 = aie::shuffle_down_fill(o4, o5, 4);

      v8 relu0 = aie::max(aie::add(out0, *(rptr++)), bfloat16(0));
      writeincr(sout, relu0);
      v8 relu1 = aie::max(aie::add(out1, *(rptr++)), bfloat16(0));
      writeincr(sout, relu1);
      v8 relu2 = aie::max(aie::add(out2, *(rptr++)), bfloat16(0));
      writeincr(sout, relu2);
    }

    if(r < 6){
      rows[0] = rows[1];
      rows[1] = rows[2];
      load_row(rows[2]);
    }
  }
}
