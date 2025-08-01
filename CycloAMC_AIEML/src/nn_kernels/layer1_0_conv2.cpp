
#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer1_0_conv2.h"
#include "fam_funcs.h"
#include "parameters.h"

/*──────────────────────── constructor ───────────────────────*/
layer1_0_conv2::layer1_0_conv2(int xoff) : m_xoff(xoff)
{
  aie::set_rounding   (aie::rounding_mode::symmetric_inf);
  aie::set_saturation (aie::saturation_mode::saturate);
}


//24*32*32=24576
//8*32*32=8192
void layer1_0_conv2::run(adf::input_buffer<bfloat16, adf::extents<1728>>& wbuf,
           input_stream<bfloat16>*                         sin ,
           adf::input_buffer<bfloat16, adf::extents<8192>>& residual,
           output_stream<bfloat16>*                        sout)
  {
    /* 型别简记 */
    using v32  = aie::vector<bfloat16,32>;
    using v64  = aie::vector<bfloat16,64>;
    using v16  = aie::vector<bfloat16,16>;
    using v8   = aie::vector<bfloat16, 8>;
    using acc16= aie::accum <accfloat ,16>;

    const v16 Z16 = aie::zeros<bfloat16,16>();
    const v32 Z32 = aie::zeros<bfloat16,32>();
    const v64 Z64 = aie::zeros<bfloat16,64>();
    auto rptr = aie::begin_restrict_vector<8>(residual);   // 8192 / 8 = 1024 个 v8


    /* ── 行缓冲：34×v8 ×3 ── */
    alignas(32) v32 row0[34], row1[34], row2[34];
    v32* rows[3] = {row0,row1,row2};

    /* 读取一行：24 次 × readincr_v<32> → 768 bf16 */
    auto load_row = [&](v32* dst){
        dst[0] = Z64.extract<32>(0);
        for(int seg=0; seg<24; ++seg){
            v32 v = readincr_v<32>(sin);          // 768 / 行
            //writeincr(sout, v);
            dst[1 + seg] = v;      
        }
        dst[25] = Z64.extract<32>(0);
    };

    /* 预热：pad 行 + 行0,1 */
    for(int i=0;i<34;++i) row0[i]=Z64.extract<32>(0);;
    load_row(row1);
    load_row(row2);
    

    /* ── 输出 32×32 ── */
    for(int r=0;r<32;++r){
      
      for(int c=0;c<32;++c){

        acc16 acc0=aie::zeros<accfloat,16>(),
              acc1=aie::zeros<accfloat,16>();

        /* --- (重置) 权重游标：按需取，不预存 --- */
        auto itap = aie::begin_restrict_vector<16>(wbuf);

        /* 3 × 8IC block */
      for (int ib = 0; ib < 3; ++ib) {

        /* 取 3 行 × 8IC 输入 */
        v32 vin[3];
        for (int kr = 0; kr < 3; ++kr) {
          v32 col32 = rows[kr][1 + c];              // 当前列
          v8  px8   = col32.extract<8>(ib);     // ic 0-7 / 8-15 / 16-23
          v16 pair  = aie::concat(px8, px8);        // 8 → 16 lane
          vin[kr].insert(0, pair);
          vin[kr].insert(1, pair);                  // 32 lane = 4×同 8IC
        }
        /* kernel-列 0..2 */
        for (int kc = 0; kc < 3; ++kc) {
          for (int krw = 0; krw < 3; ++krw) {          // 新增 krw = kernel row
            v16 lo = *(itap++);       // 8IC → OC0‑3
            v16 hi = *(itap++);       // 8IC → OC4‑7
            //v32 w  = aie::concat(lo, hi);
        
            /* OC 切半格式保持不变 */
            v32 w0 = aie::concat(lo, Z32.extract<16>(0));  // OC0‑3
            v32 w1 = aie::concat(hi, Z32.extract<16>(0));  // OC4‑7
        
            acc0 = mac_4x8_8x4(vin[krw], w0, acc0);
            acc1 = mac_4x8_8x4(vin[krw], w1, acc1);
          }
        }
      } /* ib */

      /* ── ReLU + 写回 ── */
      v8 low  = acc0.to_vector<bfloat16>().extract<8>(0);  // OC0-3
      v8 high = acc1.to_vector<bfloat16>().extract<8>(0);  // OC4-7
      v8 out  = aie::shuffle_down_fill(low, high, 4);
      v8 res   = *(rptr++);               // 顺序读取 residual 对应像素
      v8 sum   = aie::add(out, res);               // 残差相加
      v8 relu  = aie::max(sum, bfloat16(0));

      writeincr(sout, relu);              // 写出最终结果
    } /* c */

    /* ── 行滚动：再取下一行 ── */
    if (r < 30) {
      rows[0] = rows[1];
      rows[1] = rows[2];
      load_row(rows[2]);          // 读入下一原始行
    }

    
    } /* r */
}