#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer5_head.h"
#include "parameters.h"

layer5_head::layer5_head(int xoff) : m_xoff(xoff)
{
  aie::set_rounding(aie::rounding_mode::symmetric_inf);
  aie::set_saturation(aie::saturation_mode::saturate);
}

void layer5_head::run(
  adf::input_buffer<bfloat16, adf::extents<776>>& w_lin_b,
  input_stream<bfloat16>* ifm_s,
  output_stream<bfloat16>* ofm_s)
{
  using v16 = aie::vector<bfloat16, 16>;
  using acc16 = aie::accum<accfloat, 16>;

  constexpr int PIXELS = 16;
  constexpr int COUT = 8;

  const bfloat16* W = w_lin_b.data();
  const bfloat16* B = W + 96 * COUT;

  auto dot16 = [](const v16& a, const v16& b) -> float {
    v16 prod = aie::mul(a, b);
    acc16 acc;
    acc.from_vector(prod, 0);
    return aie::reduce_add(acc.to_vector<float>());
  };

  acc16 s0 = aie::zeros<accfloat,16>(),
        s1 = aie::zeros<accfloat,16>(),
        s2 = aie::zeros<accfloat,16>(),
        s3 = aie::zeros<accfloat,16>(),
        s4 = aie::zeros<accfloat,16>(),
        s5 = aie::zeros<accfloat,16>();

  for (int p = 0; p < PIXELS; ++p)
  chess_prepare_for_pipelining
  {
    v16 v0 = readincr_v<16>(ifm_s);
    v16 v1 = readincr_v<16>(ifm_s);
    v16 v2 = readincr_v<16>(ifm_s);
    v16 v3 = readincr_v<16>(ifm_s);
    v16 v4 = readincr_v<16>(ifm_s);
    v16 v5 = readincr_v<16>(ifm_s);

    s0 = aie::add(s0, acc16(v0));
    s1 = aie::add(s1, acc16(v1));
    s2 = aie::add(s2, acc16(v2));
    s3 = aie::add(s3, acc16(v3));
    s4 = aie::add(s4, acc16(v4));
    s5 = aie::add(s5, acc16(v5));
  }

  const bfloat16 scale = bfloat16(0.0625f);
  v16 gap0 = aie::mul(s0.to_vector<bfloat16>(), scale);
  v16 gap1 = aie::mul(s1.to_vector<bfloat16>(), scale);
  v16 gap2 = aie::mul(s2.to_vector<bfloat16>(), scale);
  v16 gap3 = aie::mul(s3.to_vector<bfloat16>(), scale);
  v16 gap4 = aie::mul(s4.to_vector<bfloat16>(), scale);
  v16 gap5 = aie::mul(s5.to_vector<bfloat16>(), scale);

  for (int oc = 0; oc < COUT; ++oc)
  chess_prepare_for_pipelining
  {
    const bfloat16* w_ptr = W + oc * 96;
    float acc = 0.f;

    acc += dot16(gap0, aie::load_v<16>(w_ptr +  0));
    acc += dot16(gap1, aie::load_v<16>(w_ptr + 16));
    acc += dot16(gap2, aie::load_v<16>(w_ptr + 32));
    acc += dot16(gap3, aie::load_v<16>(w_ptr + 48));
    acc += dot16(gap4, aie::load_v<16>(w_ptr + 64));
    acc += dot16(gap5, aie::load_v<16>(w_ptr + 80));

    writeincr(ofm_s, bfloat16(acc + float(B[oc])));
  }
}
