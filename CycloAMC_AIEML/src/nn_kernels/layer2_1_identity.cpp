#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer2_1_identity.h"
#include "fam_funcs.h"
#include "parameters.h"

layer2_1_identity::layer2_1_identity(int xoff) : m_xoff(xoff)
{
  aie::set_rounding(aie::rounding_mode::symmetric_inf);
  aie::set_saturation(aie::saturation_mode::saturate);
}

void layer2_1_identity::run(
  input_stream<bfloat16>* sin,
  adf::output_buffer<bfloat16, adf::extents<4096>>& buf)
{
  using v32 = aie::vector<bfloat16, 32>;
  auto it = aie::begin_restrict_vector<32>(buf);

  for (int i = 0; i < 4096 / 32; ++i) {
    v32 v = readincr_v<32>(sin);
    *(it++) = v;
  }
}
