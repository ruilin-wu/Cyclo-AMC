
#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "layer0_stem.h"
#include "fam_funcs.h"
#include "parameters.h"

/*──────────────────────── constructor ───────────────────────*/
layer0_stem::layer0_stem(int xoff) : m_xoff(xoff)
{
  aie::set_rounding   (aie::rounding_mode::symmetric_inf);
  aie::set_saturation (aie::saturation_mode::saturate);
}


void layer0_stem::run(adf::input_buffer<bfloat16, adf::extents<72>>& wbuf,
                      input_stream<bfloat16>*                        sin ,
                      output_stream<bfloat16>*                       bout )
{
  using v16   = aie::vector<bfloat16,16>;
  using v32   = aie::vector<bfloat16,32>;
  using v8   = aie::vector<bfloat16,8>;
  
  using acc16 = aie::accum <accfloat ,16>;

  /*── 1. 临时缓冲定义 ───────────────────────────────*/
  std::array<v32,3>  buffd;        // 3   行输入，每行 32 像素
  std::array<v32,3>  buffw;        // 3×tap 权重（上 16 lane 0）
  std::array<acc16,3> acc;         // 3   块累加器
  const v32 Z32 = aie::zeros<bfloat16,32>();

  /*── 2. 权重迭代器（16 lane / tap）─────────────────*/
  auto itap = aie::begin_restrict_vector<8>(wbuf);   // 9 taps × 16 ch = 144

  /*── 3. 三行环形缓冲（66 = 64 + 两侧 pad）──────────*/
  alignas(32) bfloat16 row0[66], row1[66], row2[66],tmp[66];
  
  auto load_row = [&](bfloat16* dst){
      dst[0]  = 0;
      aie::vector<bfloat16,64> v = readincr_v<64>(sin);
      v.store(dst + 1);                            // 写满 64 个像素
      dst[65] = 0;
  };
  /* 顶部 pad 行 = 全 0，直接 memset 为 0 */
  for(int i=0;i<66;++i) row0[i]=bfloat16(0);
  load_row(row1);               // 第 0 行
  load_row(row2);               // 第 1 行

  /* 用指针数组实现“行滚动”，避免 std::copy_n */
  bfloat16* rows[3] = { row0, row1, row2 };
    const v16 Z16 = aie::zeros<bfloat16,16>();
  /*── 4. 遍历 64 行输出 ─────────────────────────────*/
  for (unsigned rr=0; rr<64; ++rr)
    
  {
    bool bottom_row = (rr == 63);

    /* 将三行按段(32 px)搬入 buffd ─────────────────*/
    for (unsigned seg=0; seg<4; ++seg)
    {           
        int base = seg*16;                 // 段起始列 0/32
        int base2 = base+16;                 // 段起始列 0/32
        /* row_prev / curr / next → buffd[0/1/2] */
        if (seg==3){
            buffd[2].insert(1, Z16);
            buffd[0].insert(1, Z16);
            buffd[1].insert(1, Z16);            
            }
        else {
            buffd[2].insert(1, aie::load_v<16>(rows[2]+base2));
            buffd[0].insert(1, aie::load_v<16>(rows[0]+base2));
            buffd[1].insert(1, aie::load_v<16>(rows[1]+base2));            
            } 
        buffd[0].insert(0, aie::load_v<16>(rows[0]+base));
        buffd[1].insert(0, aie::load_v<16>(rows[1]+base));
        if (bottom_row){
            buffd[2].insert(0, Z32);}    // 最后一行下侧 pad
        else 
        {   buffd[2].insert(0, aie::load_v<16>(rows[2]+base));                       
        }
                
        /* 列滑窗：一次 2 列，共 16 次 */
        unsigned c = 0;
        for (unsigned c=0; c<16; ++c) 
        //一次 writeincr 就写出了 16 个通道在 (row rr, col c) 这一个输出像素的值。      
        {
            /*―― 第 1 行：tap 0 1 2 ――*/
            buffw[0] = Z32;  buffw[1] = Z32;  buffw[2] = Z32;
            buffw[0].insert(0, *(itap+0));
            buffw[1].insert(0, *(itap+1));
            buffw[2].insert(0, *(itap+2));
            acc[0] = mul_elem_16_2(aie::broadcast<bfloat16,32>(buffd[0].get(c  )), buffw[0]);
            acc[1] = mul_elem_16_2(aie::broadcast<bfloat16,32>(buffd[0].get(c+1)), buffw[1]);
            acc[2] = mul_elem_16_2(aie::broadcast<bfloat16,32>(buffd[0].get(c+2)), buffw[2]);

            /*―― 第 2 行：tap 3 4 5 ――*/
            buffw[0].insert(0, *(itap+3));
            buffw[1].insert(0, *(itap+4));
            buffw[2].insert(0, *(itap+5));
            acc[0] = mac_elem_16_2(aie::broadcast<bfloat16,32>(buffd[1].get(c  )), buffw[0], acc[0]);
            acc[1] = mac_elem_16_2(aie::broadcast<bfloat16,32>(buffd[1].get(c+1)), buffw[1], acc[1]);
            acc[2] = mac_elem_16_2(aie::broadcast<bfloat16,32>(buffd[1].get(c+2)), buffw[2], acc[2]);

            /*―― 第 3 行：tap 6 7 8 ――*/
            buffw[0].insert(0, *(itap+6));
            buffw[1].insert(0, *(itap+7));
            buffw[2].insert(0, *(itap+8));
            acc[0] = mac_elem_16_2(aie::broadcast<bfloat16,32>(buffd[2].get(c  )), buffw[0], acc[0]);
            acc[1] = mac_elem_16_2(aie::broadcast<bfloat16,32>(buffd[2].get(c+1)), buffw[1], acc[1]);
            acc[2] = mac_elem_16_2(aie::broadcast<bfloat16,32>(buffd[2].get(c+2)), buffw[2], acc[2]);

            /*―― 汇总 & ReLU & 输出 ――
            acc16 sum = aie::add(aie::add(acc[0],acc[1]), acc[2]);
            v16  vout = aie::max(sum.to_vector<bfloat16>(), bfloat16(0));
            //v16  vout = sum.to_vector<bfloat16>(), bfloat16(0);
            writeincr(bout, vout);
            
            v16 sum16 = aie::add(aie::add(acc[0],acc[1]), acc[2]).to_vector<bfloat16>();
            v8  out8  = sum16.extract<8>(0);                 // 只保留低 8 lane
            out8      = aie::max(out8,bfloat16(0));          // ReLU
            writeincr(bout,out8);
            */

            v8 out8 = aie::max( aie::add(aie::add(acc[0], acc[1]), acc[2])
                      .to_vector<bfloat16>()
                      .extract<8>(0),    // 低 8 lane
                    bfloat16(0) );                // ReLU

            writeincr(bout,out8);
            
        }      
    }

     //行滚动：指针旋转，无 std::copy_n ――――――――――――――――― 
        if (rr < 62) {                       // 0-61 真正滚动        
        rows[0] = rows[1];
        rows[1] = rows[2]; 
        load_row(rows[2]);               // 拉下一行输入
        }
  }
}


  