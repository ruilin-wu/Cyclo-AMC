
#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer1_0_down.h"
#include "fam_funcs.h"
#include "parameters.h"

/*──────────────────────── constructor ───────────────────────*/
layer1_0_down::layer1_0_down(int xoff) : m_xoff(xoff)
{
  aie::set_rounding   (aie::rounding_mode::symmetric_inf);
  aie::set_saturation (aie::saturation_mode::saturate);
}


//24*32*32=24576
//8*32*32=8192
void layer1_0_down::run(adf::input_buffer<bfloat16, adf::extents<1728>>& wbuf,
           input_stream<bfloat16>*                                   sin,
           adf::output_buffer<bfloat16, adf::extents<8192>>&         buf)
  {
    using v32 = aie::vector<bfloat16, 32>;

    /* 取得一个可写游标：指向 buf 起始处，步长 = 32 lane */
    auto it = aie::begin_restrict_vector<32>(buf);

    /* 连续读取 256 次，每次一个 v32，然后写入 buf */
    for (int i = 0; i < 256; ++i) {
      v32 v = readincr_v<32>(sin);   // 从输入流消费 32 个 bf16
      *(it++) = v;                   // 顺序写到 buf
    }
  }
