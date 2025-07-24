
#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer0_stem_maxpool.h"
#include "fam_funcs.h"
#include "parameters.h"

/*──────────────────────── constructor ───────────────────────*/
layer0_stem_maxpool::layer0_stem_maxpool(int xoff) : m_xoff(xoff)
{
  aie::set_rounding   (aie::rounding_mode::symmetric_inf);
  aie::set_saturation (aie::saturation_mode::saturate);
}


/*──────────────── MaxPool K3 S2 P1 ───────────*/
void layer0_stem_maxpool::run(input_stream<bfloat16>*  sin,
                              adf::output_buffer<bfloat16,
                                  adf::extents<8192>>& __restrict out_buf)
{
  using v8 = aie::vector<bfloat16, 8>;          // 1 像素 = 8 通道
  const v8 Z = aie::zeros<bfloat16, 8>();

  /* —1— 三个物理行缓冲（66 像素，含左右 pad） */
  alignas(32) v8 row0[66], row1[66], row2[66];
  v8* rows[3] = {row0, row1, row2};             // rows[0]=top , rows[2]=bottom

  auto load_row = [&](v8* dst) {
      dst[0] = Z;                               // left pad
      for (int c = 0; c < 64; ++c)
          dst[c + 1] = readincr_v<8>(sin);      // 64 像素 × 8ch
      dst[65] = Z;                              // right pad
  };

  /* ── ① 预热：top-pad 行 + 实际行 0,1 ── */
  for (int i = 0; i < 66; ++i) row0[i] = Z;     // row-1 (pad)
  load_row(row1);                               // 行 0
  load_row(row2);                               // 行 1   ← 目前共 2 次 load_row()

  auto itw = aie::begin_restrict_vector<8>(out_buf);

  /* ── ② 输出 32 × 32 像素 ── */
  for (int r_out = 0; r_out < 32; ++r_out)
    chess_prepare_for_pipelining
  {
    for (int c_out = 0; c_out < 32; ++c_out)
      chess_prepare_for_pipelining
    {
      int c0 = c_out * 2;                       // 窗口左上角

      v8 p00 = rows[0][c0],   p01 = rows[0][c0+1], p02 = rows[0][c0+2];
      v8 p10 = rows[1][c0],   p11 = rows[1][c0+1], p12 = rows[1][c0+2];
      v8 p20 = rows[2][c0],   p21 = rows[2][c0+1], p22 = rows[2][c0+2];

      v8 m0  = aie::max(aie::max(p00,p01), p02);
      v8 m1  = aie::max(aie::max(p10,p11), p12);
      v8 m2  = aie::max(aie::max(p20,p21), p22);
      v8 out = aie::max(aie::max(m0 ,m1 ), m2); // 3×3 池化

      *itw++ = out;          // 顺序写入缓冲，每次 8 通道
    }

    /* ── ③ 行滚动：滑下 2 行 ⇒ 新读 2 行 ──
           rows: [top, mid, bot] → [old_bot, new0, new1]            */
    if (r_out < 31)                             // 共执行 31 次
    {
      v8* old_top = rows[0];
      rows[0] = rows[2];                        // 旧 bottom → 新 top
      rows[2] = old_top;                        // 回收旧 top 作新 bottom

      load_row(rows[1]);                        // 读下一真实行
      load_row(rows[2]);                        // 再读下一真实行
      /*  每轮两次 load_row → 2 × 31 = 62
          加上启动 2 次，共 64 次，刚好 64 行输入 */
    }
  }
}
