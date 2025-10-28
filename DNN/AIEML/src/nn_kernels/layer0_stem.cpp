#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer0_stem.h"
#include "fam_funcs.h"
#include "parameters.h"

layer0_stem::layer0_stem(int xoff) : m_xoff(xoff)
{
  aie::set_rounding(aie::rounding_mode::symmetric_inf);
  aie::set_saturation(aie::saturation_mode::saturate);
}

void layer0_stem::run(adf::input_buffer<bfloat16, adf::extents<72>>& wbuf,
                      input_stream<bfloat16>* sin,
                      output_stream<bfloat16>* bout)
{
  using v16 = aie::vector<bfloat16,16>;
  using v32 = aie::vector<bfloat16,32>;
  using v8 = aie::vector<bfloat16,8>;
  using acc16 = aie::accum<accfloat,16>;

  std::array<v32,3> buffd;
  std::array<v32,3> buffw;
  std::array<acc16,3> acc;
  const v32 Z32 = aie::zeros<bfloat16,32>();

  auto itap = aie::begin_restrict_vector<8>(wbuf);

  alignas(32) bfloat16 row0[66], row1[66], row2[66], tmp[66];

  auto load_row = [&](bfloat16* dst){
    dst[0] = 0;
    aie::vector<bfloat16,64> v = readincr_v<64>(sin);
    v.store(dst + 1);
    dst[65] = 0;
  };

  for(int i=0;i<66;++i) row0[i]=bfloat16(0);
  load_row(row1);
  load_row(row2);

  bfloat16* rows[3] = { row0, row1, row2 };
  const v16 Z16 = aie::zeros<bfloat16,16>();

  for (unsigned rr=0; rr<64; ++rr)
  {
    bool bottom_row = (rr == 63);

    for (unsigned seg=0; seg<4; ++seg)
    {
      int base = seg*16;
      int base2 = base+16;

      if (seg==3){
        buffd[2].insert(1, Z16);
        buffd[0].insert(1, Z16);
        buffd[1].insert(1, Z16);
      } else {
        buffd[2].insert(1, aie::load_v<16>(rows[2]+base2));
        buffd[0].insert(1, aie::load_v<16>(rows[0]+base2));
        buffd[1].insert(1, aie::load_v<16>(rows[1]+base2));
      }

      buffd[0].insert(0, aie::load_v<16>(rows[0]+base));
      buffd[1].insert(0, aie::load_v<16>(rows[1]+base));
      if (bottom_row){
        buffd[2].insert(0, Z32);
      } else {
        buffd[2].insert(0, aie::load_v<16>(rows[2]+base));
      }

      unsigned c = 0;
      for (unsigned c=0; c<16; ++c)
      {
        buffw[0] = Z32; buffw[1] = Z32; buffw[2] = Z32;
        buffw[0].insert(0, *(itap+0));
        buffw[1].insert(0, *(itap+1));
        buffw[2].insert(0, *(itap+2));
        acc[0] = mul_elem_16_2(aie::broadcast<bfloat16,32>(buffd[0].get(c)), buffw[0]);
        acc[1] = mul_elem_16_2(aie::broadcast<bfloat16,32>(buffd[0].get(c+1)), buffw[1]);
        acc[2] = mul_elem_16_2(aie::broadcast<bfloat16,32>(buffd[0].get(c+2)), buffw[2]);

        buffw[0].insert(0, *(itap+3));
        buffw[1].insert(0, *(itap+4));
        buffw[2].insert(0, *(itap+5));
        acc[0] = mac_elem_16_2(aie::broadcast<bfloat16,32>(buffd[1].get(c)), buffw[0], acc[0]);
        acc[1] = mac_elem_16_2(aie::broadcast<bfloat16,32>(buffd[1].get(c+1)), buffw[1], acc[1]);
        acc[2] = mac_elem_16_2(aie::broadcast<bfloat16,32>(buffd[1].get(c+2)), buffw[2], acc[2]);

        buffw[0].insert(0, *(itap+6));
        buffw[1].insert(0, *(itap+7));
        buffw[2].insert(0, *(itap+8));
        acc[0] = mac_elem_16_2(aie::broadcast<bfloat16,32>(buffd[2].get(c)), buffw[0], acc[0]);
        acc[1] = mac_elem_16_2(aie::broadcast<bfloat16,32>(buffd[2].get(c+1)), buffw[1], acc[1]);
        acc[2] = mac_elem_16_2(aie::broadcast<bfloat16,32>(buffd[2].get(c+2)), buffw[2], acc[2]);

        v8 out8 = aie::max(
          aie::add(aie::add(acc[0], acc[1]), acc[2])
          .to_vector<bfloat16>()
          .extract<8>(0),
          bfloat16(0)
        );
        writeincr(bout,out8);
      }
    }

    if (rr < 62) {
      rows[0] = rows[1];
      rows[1] = rows[2];
      load_row(rows[2]);
    }
  }
}
