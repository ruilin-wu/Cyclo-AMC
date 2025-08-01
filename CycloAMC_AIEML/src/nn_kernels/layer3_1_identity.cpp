#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer3_1_identity.h"
#include "fam_funcs.h"
#include "parameters.h"

layer3_1_identity::layer3_1_identity(int xoff) : m_xoff(xoff)
{
  aie::set_rounding(aie::rounding_mode::symmetric_inf);
  aie::set_saturation(aie::saturation_mode::saturate);
}

void layer3_1_identity::run(
  input_stream<bfloat16>* sin,
  adf::output_buffer<bfloat16, adf::extents<1536>>& buf)
{
  using v32 = aie::vector<bfloat16, 32>;
  auto it = aie::begin_restrict_vector<32>(buf);

  for (int i = 0; i < 1536 / 32; ++i) {
    v32 v = readincr_v<32>(sin);
    *(it++) = v;
  }
}
