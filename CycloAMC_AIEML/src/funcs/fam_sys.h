#ifndef __SSCA_SYS_H__
#define __SSCA_SYS_H__
#
#include <adf.h>



#include "parameters.h"
#include "fam_stage1.h"
#include "fam_stage2.h"
#include "fam_to_nn.h"
#include "layer0_stem.h"
#include "broad_cast.h"
#include "layer0_stem_maxpool.h"
#include "layer1_0_conv1.h"
#include "layer1_0_down.h"
#include "layer1_0_conv2.h"
#include "layer2_0_conv1.h"
#include "layer2_0_down.h"
#include "layer2_0_conv2.h"
#include "layer2_1_identity.h"
#include "layer2_1_conv1.h"
#include "layer2_1_conv2.h"
#include "fft64_test.h"
#include "layer3_0_down.h"
#include "layer3_0_conv1.h"
#include "layer3_0_conv2.h"
#include "layer3_1_identity.h"
#include "layer3_1_conv1.h"
#include "layer3_1_conv2.h"

#include "layer4_0_down.h"
#include "layer4_0_conv1.h"
#include "layer4_0_conv2.h"
#include "layer4_1_identity.h"
#include "layer4_1_conv1.h"
#include "layer4_1_conv2.h"
#include "layer5_head.h"
#include "fam_stage2_1.h"
#include "fam_stage2_2.h"
/*
#include "fam_stage2.h"
#include "conv_stage1.h"
#include "conv_stage2.h"
#include "norm.h"
#include "channel1.h"
#include "channel2.h"
*/
using namespace adf;


template<int xoff, int yoff, int index>
    class mem_to_4_graph_1 : public graph {
    private:
        kernel                   k_fam[4];
        //kernel                   k_conv;
        
        shared_buffer<cbfloat16> in_mem;
        //shared_buffer<cbfloat16> out_mem;

    public:
        port<input>  din;
        port<output> dout[4];
        mem_to_4_graph_1()
        {   
            /* ① 4 个 kernel */
            
            for (int i = 0; i < 4; ++i) {
                k_fam[i] = kernel::create_object<fam_stage1>(i);
                source   (k_fam[i])      = "fam_stage1.cpp";
                runtime<ratio>(k_fam[i]) = 0.9;               
            }
            

            location<kernel>(k_fam[0]) = tile(xoff + 0, yoff + 0); 
            location<buffer>(k_fam[0].out[0]) = {bank(xoff + 0, yoff + 0, 2), bank(xoff + 0, yoff + 0, 3)};
            location<stack >(k_fam[0]) = location<kernel>(k_fam[0]);
            location<buffer>(k_fam[0].in[0]) = location<kernel>(k_fam[0]);

            location<kernel>(k_fam[1]) = tile(xoff + 1, yoff + 1); 
            location<buffer>(k_fam[1].out[0]) = {bank(xoff + 1, yoff + 1, 2), bank(xoff + 1, yoff + 1, 3)};
            location<stack >(k_fam[1]) = bank(xoff + 0, yoff + 1, 1);
            location<buffer>(k_fam[1].in[0]) = {bank(xoff + 0, yoff + 1, 2), bank(xoff + 0, yoff + 1, 3)};

            location<kernel>(k_fam[2]) = tile(xoff + 2, yoff + 1); 
            location<buffer>(k_fam[2].out[0]) = {bank(xoff + 1, yoff + 1, 0), bank(xoff + 1, yoff + 1, 1)};
            location<stack >(k_fam[2]) = location<kernel>(k_fam[2]);
            location<buffer>(k_fam[2].in[0]) = location<kernel>(k_fam[2]);

            location<kernel>(k_fam[3]) = tile(xoff + 2, yoff + 0); 
            location<buffer>(k_fam[3].out[0]) = {bank(xoff + 1, yoff + 0, 2), bank(xoff + 1, yoff + 0, 3)};
            location<stack >(k_fam[3]) = location<kernel>(k_fam[3]);
            location<buffer>(k_fam[3].in[0]) = location<kernel>(k_fam[3]);
            
        /* ② 共享缓冲区：1 写 / 4 读 */
        in_mem = shared_buffer<cbfloat16>::create({1, 1, 560}, 1, 4);
        location<buffer>(in_mem) = location<kernel>(k_fam[0])    + relative_offset({ .col_offset = 0 });  // 只支持列偏移
        num_buffers(in_mem) = 2;

        /* ③ PL → in_mem */
        write_access(in_mem.in[0]) =
            tiling({ .buffer_dimension = {560},
                     .tiling_dimension = {560},
                     .offset           = {0},
                     .tile_traversal   = {{0,0,1} } });
        connect(din, in_mem.in[0]);

        /* ④ in_mem → kernels（偏移 0/512/1024/1536，各 704 点） */
        const int off[4] = {0, 128, 256, 384};
        for (int i = 0; i < 4; ++i) {
            read_access(in_mem.out[i]) =
                tiling({ .buffer_dimension = {560},
                         .tiling_dimension = {176},
                         .offset           = {off[i]},
                         .tile_traversal   = { {0,0,1} } });
            connect(in_mem.out[i], k_fam[i].in[0]);        
        }
        /* ⑤  k_conv → 图外输出*/ 
        connect(k_fam[0].out[0], dout[0]);
        connect(k_fam[1].out[0], dout[1]);
        connect(k_fam[2].out[0], dout[2]);
        connect(k_fam[3].out[0], dout[3]);
        
    }
};




template<int xoff, int yoff, int base_index>
class k_8_stage2_test : public graph {
private:
    static constexpr int NumKernels = 8; // 
    kernel k_stage2_1[4];
    kernel k_stage2_2[4];
public:
    port<input>  din;
    port<output> dout[1];
    shared_buffer<bfloat16> out_mem;
    

    k_8_stage2_test() {
        for (int i = 0; i < 4; ++i) {
            k_stage2_1[i] = kernel::create_object<fam_stage2_1>(base_index + 4*i);
            source   (k_stage2_1[i])      = "fam_stage2_1.cpp";
            runtime<ratio>(k_stage2_1[i]) = 0.9;
            connect(din, k_stage2_1[i].in[0]);            
        }
        for (int i = 0; i < 4; ++i) {
            k_stage2_2[i] = kernel::create_object<fam_stage2_2>(base_index + 4*i+2);
            source   (k_stage2_2[i])      = "fam_stage2_2.cpp";
            runtime<ratio>(k_stage2_2[i]) = 0.9;
            connect(din, k_stage2_2[i].in[0]);            
        }

        location<kernel>(k_stage2_1[0])        = tile(xoff+0, yoff+0);
        location<kernel>(k_stage2_2[0])        = tile(xoff+0, yoff+1);
        location<kernel>(k_stage2_1[1])        = tile(xoff+0, yoff+2);
        location<kernel>(k_stage2_2[1])        = tile(xoff+0, yoff+3);
        location<kernel>(k_stage2_1[2])        = tile(xoff+0, yoff+4);
        location<kernel>(k_stage2_2[2])        = tile(xoff+0, yoff+5);
        location<kernel>(k_stage2_1[3])        = tile(xoff+0, yoff+6);
        location<kernel>(k_stage2_2[3])        = tile(xoff+0, yoff+7);  
        
        
        for (int i = 0; i < 4; ++i) {           
            location<buffer>(k_stage2_2[i].out[0]) = location<kernel>(k_stage2_2[i]);
            location<stack >(k_stage2_2[i])        = location<kernel>(k_stage2_2[i]);
        }

        for (int i = 0; i < 4; ++i) {                       
            location<stack >(k_stage2_1[i])        = location<kernel>(k_stage2_1[i]);
        }


        /* ──────────────────────────────
        * K_2 → K_1
        * ──────────────────────────────*/
        for (int i = 0; i < 4; ++i) {                       
            connect(k_stage2_2[i].out[0], k_stage2_1[i].in[1]);
        }



        /* ──────────────────────────────
        * out_mem → dout  整块 8192 顺序读
        * ──────────────────────────────*/
        out_mem = shared_buffer<bfloat16>::create({1, 1, 256*4}, 4, 1);
        num_buffers(out_mem) = 2;
        location<buffer>(out_mem) = location<kernel>(k_stage2_1[0])    + relative_offset({ .col_offset = 0 });  // 只支持列偏移

        for (int i = 0; i < 4; ++i) {
            write_access(out_mem.in[i]) =
                tiling({
                    .buffer_dimension = {1, 1, 256*4},   // 整块缓冲
                    .tiling_dimension = {1, 1, 256},      // 一次写 8 点
                    .offset           = {0, 0, 256 * i},  // 各核起始 0/8/16/24
                    .tile_traversal   = {
                        { .dimension = 2, .stride = 256, .wrap = 1 }, // 交织步长 32
                        { .dimension = 0, .stride = 1,  .wrap = 1 },
                        { .dimension = 1, .stride = 1,  .wrap = 1 }
                    }
                });
            connect(k_stage2_1[i].out[0], out_mem.in[i]);
            }

        /* ──────────────────────────────
        * out_mem → dout  整块 8192 顺序读
        * ──────────────────────────────*/
        read_access(out_mem.out[0]) =
            tiling({
                .buffer_dimension = {1,1,256*4},
                .tiling_dimension = {1,1,256*4},
                .offset           = {0,0,0},
                .tile_traversal   = { {2,256*4,1},{0,1,1},{1,1,1} }
            });
        connect(out_mem.out[0], dout[0]);
    }
};


template<int xoff, int yoff, int base_index>
class k_4_stage2_test : public graph {
private:
    static constexpr int NumKernels = 8; // 
    
    kernel k_stage2_2[4];
    //kernel k_stage2_1[6];
    //kernel k_stage2_3[5];
public:
    port<input>  din;
    port<output> dout[1];
    shared_buffer<bfloat16> out_mem;
    

    k_4_stage2_test() {
        for (int i = 0; i < 4; ++i) {
            k_stage2_2[i] = kernel::create_object<fam_stage2_2>(base_index+4*i);
            source   (k_stage2_2[i])      = "fam_stage2_2.cpp";
            runtime<ratio>(k_stage2_2[i]) = 0.9;
            connect(din, k_stage2_2[i].in[0]);
            location<kernel>(k_stage2_2[i]) = tile(xoff + 0, yoff + i); 
            location<buffer>(k_stage2_2[i].out[0]) = location<kernel>(k_stage2_2[i]);
            location<stack >(k_stage2_2[i])        = location<kernel>(k_stage2_2[i]);          
        }

        /* ──────────────────────────────
        * out_mem → dout  整块 8192 顺序读
        * ──────────────────────────────*/
       out_mem = shared_buffer<bfloat16>::create({1, 1, 256*4}, 4, 1);
       num_buffers(out_mem) = 2;
       location<buffer>(out_mem) = location<kernel>(k_stage2_2[0])    + relative_offset({ .col_offset = 0 });  // 只支持列偏移

       for (int i = 0; i < 4; ++i) {
           write_access(out_mem.in[i]) =
               tiling({
                   .buffer_dimension = {1, 1, 256*4},   // 整块缓冲
                   .tiling_dimension = {1, 1, 256},      // 一次写 8 点
                   .offset           = {0, 0, 256 * i},  // 各核起始 0/8/16/24
                   .tile_traversal   = {
                       { .dimension = 2, .stride = 256, .wrap = 1 }, // 交织步长 32
                       { .dimension = 0, .stride = 1,  .wrap = 1 },
                       { .dimension = 1, .stride = 1,  .wrap = 1 }
                   }
               });
           connect(k_stage2_2[i].out[0], out_mem.in[i]);
           }

       /* ──────────────────────────────
       * out_mem → dout  整块 8192 顺序读
       * ──────────────────────────────*/
       read_access(out_mem.out[0]) =
           tiling({
               .buffer_dimension = {1,1,256*4},
               .tiling_dimension = {1,1,256*4},
               .offset           = {0,0,0},
               .tile_traversal   = { {2,256*4,1},{0,1,1},{1,1,1} }
           });
       connect(out_mem.out[0], dout[0]);

    }
} ;       



template<
    int  X0  = 16, int  Y0  = 3,            // mem_to_4 起点
    int  X1  = 0, int  Y1  = 0,            // stage-2  起点
    int  NK  = 1                            // stage-2 个数
>
class fam_top : public adf::graph
{
private:
    static constexpr int NumKernels = 4; // 用于替换所有的 16
    kernel                   k_broad_cast;
    mem_to_4_graph_1<X0, Y0,   0>  stage1;   // 一输入一输出
    
    k_4_stage2_test<X1+ 0, Y1, 1>   stage2_0_15;     
    k_4_stage2_test<X1+ 1, Y1, 17>   stage2_16_31;  
    k_4_stage2_test<X1+ 2, Y1, 33>   stage2_32_47;  
    k_4_stage2_test<X1+ 3, Y1, 49>   stage2_48_63;  
    
public:
    /*** 顶层端口：一进一出 ***/
    adf::port<input>  input;
    adf::port<output> output[1];
    shared_buffer<bfloat16> scd_mem;
    fam_top()
    {       
        connect<>(input      , stage1.din);        
        k_broad_cast = kernel::create_object<broad_cast>(0);          
        source(k_broad_cast) = "broad_cast.cpp";
        runtime<ratio>(k_broad_cast) = 0.9;
        location<kernel>(k_broad_cast) = tile(X0 + 1, Y0 + 0); 
        location<stack >(k_broad_cast) = location<kernel>(k_broad_cast);
        connect<>(stage1.dout[0], k_broad_cast.in[0]);
        connect<>(stage1.dout[1], k_broad_cast.in[1]);
        connect<>(stage1.dout[2], k_broad_cast.in[2]);
        connect<>(stage1.dout[3], k_broad_cast.in[3]);

        connect<>(k_broad_cast.out[0],stage2_0_15.din );
        connect<>(k_broad_cast.out[0], stage2_16_31.din );
        connect<>(k_broad_cast.out[0], stage2_32_47.din );
        connect<>(k_broad_cast.out[0], stage2_48_63.din );

        /* ──────────────────────────────
        * scd_mem → dout  整块 8192 顺序读
        * ──────────────────────────────*/
       scd_mem = shared_buffer<bfloat16>::create({1, 1, 128*NumKernels*8}, NumKernels, 1);
       num_buffers(scd_mem) = 2;
       location<buffer>(scd_mem) = location<kernel>(k_broad_cast)    + relative_offset({ .col_offset = 0 });  // 只支持列偏移

       for (int i = 0; i < NumKernels; ++i) {
           write_access(scd_mem.in[i]) =
               tiling({
                   .buffer_dimension = {1, 1, 128*NumKernels*8},   // 整块缓冲
                   .tiling_dimension = {1, 1, 128*8},      // 一次写 8 点
                   .offset           = {0, 0, 128 * i*8},  // 各核起始 0/8/16/24
                   .tile_traversal   = {
                       { .dimension = 2, .stride = 128*8, .wrap = 1 }, // 交织步长 32
                       { .dimension = 0, .stride = 1,  .wrap = 1 },
                       { .dimension = 1, .stride = 1,  .wrap = 1 }
                   }
               });
           }
       connect<>(stage2_0_15.dout[0] , scd_mem.in[0]);
       connect<>(stage2_16_31.dout[0] , scd_mem.in[1]);       
       connect<>(stage2_32_47.dout[0] , scd_mem.in[2]);
       connect<>(stage2_48_63.dout[0] , scd_mem.in[3]);
       
       /* ──────────────────────────────
       * out_mem → dout  整块 8192 顺序读
       * ──────────────────────────────*/
       read_access(scd_mem.out[0]) =
           tiling({
               .buffer_dimension = {1,1,128*8*NumKernels},
               .tiling_dimension = {1,1,128*8*NumKernels},
               .offset           = {0,0,0},
               .tile_traversal   = { {2,128*8*NumKernels,1},{0,1,1},{1,1,1} }
           });
       connect(scd_mem.out[0], output[0]);
        
 
    }
};


template<
    int  X0  = 16, int  Y0  = 3,            // mem_to_4 起点
    int  X1  = 0, int  Y1  = 0,            // stage-2  起点
    int  NK  = 1                            // stage-2 个数
>
class fam_top_stem : public adf::graph
{
private:
    static constexpr int NumKernels = 4; // 用于替换所有的 16
    kernel                   k_broad_cast;
    mem_to_4_graph_1<X0, Y0,   0>  stage1;   // 一输入一输出    
    k_8_stage2_test<X1+ 0, Y1, 1>   stage2_0_15;     
    k_8_stage2_test<X1+ 2, Y1, 17>   stage2_16_31;  
    k_8_stage2_test<X1+ 4, Y1, 33>   stage2_32_47;  
    k_8_stage2_test<X1+ 6, Y1, 49>   stage2_48_63;  
    

public:
    /*** 顶层端口：一进一出 ***/
    adf::port<input>  input;
    adf::port<output> output[3];
    shared_buffer<bfloat16> scd_mem;
    fam_top_stem()
    {       
        connect<>(input      , stage1.din);        
        k_broad_cast = kernel::create_object<broad_cast>(0);          
        source(k_broad_cast) = "broad_cast.cpp";
        runtime<ratio>(k_broad_cast) = 0.9;
        location<kernel>(k_broad_cast) = tile(X0 + 1, Y0 + 0); 
        location<stack >(k_broad_cast) = location<kernel>(k_broad_cast);
        connect<>(stage1.dout[0], k_broad_cast.in[0]);
        connect<>(stage1.dout[1], k_broad_cast.in[1]);
        connect<>(stage1.dout[2], k_broad_cast.in[2]);
        connect<>(stage1.dout[3], k_broad_cast.in[3]);

        connect<>(k_broad_cast.out[0],stage2_0_15.din );
        connect<>(k_broad_cast.out[0], stage2_16_31.din );
        connect<>(k_broad_cast.out[0], stage2_32_47.din );
        connect<>(k_broad_cast.out[0], stage2_48_63.din );

        /* ──────────────────────────────
        * scd_mem → dout  整块 8192 顺序读
        * ──────────────────────────────*/
       scd_mem = shared_buffer<bfloat16>::create({64, 64, 1}, 4, 3);
       num_buffers(scd_mem) = 2;
       location<buffer>(scd_mem) = location<kernel>(k_broad_cast)    + relative_offset({ .col_offset = 5 });  // 只支持列偏移

       for (int i = 0; i <4; ++i) {
           write_access(scd_mem.in[i]) =
               tiling({
                   .buffer_dimension = {64, 64, 1},   // 整块缓冲
                   .tiling_dimension = {64, 16, 1},      // 一次写 8 点
                   .offset           = {0, 16*i, 0},  // 各核起始 0/8/16/24
                   .tile_traversal   = {
                       { .dimension = 2, .stride = 1, .wrap = 1 }, // 交织步长 32
                       { .dimension = 0, .stride = 1,  .wrap = 1 },
                       { .dimension = 1, .stride = 1,  .wrap = 1 }
                   }
               });
           }
       connect<>(stage2_0_15.dout[0] , scd_mem.in[0]);
       connect<>(stage2_16_31.dout[0] , scd_mem.in[1]);       
       connect<>(stage2_32_47.dout[0] , scd_mem.in[2]);
       connect<>(stage2_48_63.dout[0] , scd_mem.in[3]);
       
       /* ──────────────────────────────
       * out_mem → dout  整块 8192 顺序读
       * ──────────────────────────────*/
      for (int i = 0; i <3; ++i) {
            read_access(scd_mem.out[i]) = tiling({
                // 1. 原始数据区：64 × 64 × 8 
                .buffer_dimension   = {64, 64, 1},
                .tiling_dimension   = {64, 64, 1},
                .offset             = {0, 0,  0},
                .tile_traversal     = {
                    { .dimension = 0, .stride = 0, .wrap = 1 },  // W 维
                    { .dimension = 1, .stride = 0, .wrap = 1 },  // H 维
                    { .dimension = 2, .stride =  0, .wrap = 1 }   // C 维
                }
            });
       connect(scd_mem.out[i], output[i]);
       }

    }
};



template<
    int  X0  = 16, int  Y0  = 3,            // mem_to_4 起点
    int  X1  = 0, int  Y1  = 0,            // stage-2  起点
    int  NK  = 1                            // stage-2 个数
>
class stem_alter : public adf::graph
{
private:
    static constexpr int NuminKernels = 1; // 用于替换所有的 16
    static constexpr int NumoutKernels = 1; // 用于替换所有的 16
    kernel                   k_stem[3];
    kernel                   k_stem_max[3];
    
public:
    /*** 顶层端口：一进一出 ***/
    adf::port<input>  input_image[3];
    adf::port<input>  input_max[1];
    
    adf::port<input>  input_weight[1];
    adf::port<output> output[1];
    //shared_buffer<bfloat16> stem_mem;
    shared_buffer<bfloat16> weight_mem;
    shared_buffer<bfloat16> stemout_mem;


    stem_alter()
    {   
        for (int i = 0; i < 3; ++i) {
            k_stem_max[i] = kernel::create_object<layer0_stem_maxpool>(0);          
            source(k_stem_max[i]) = "layer0_stem_maxpool.cpp";
            runtime<ratio>(k_stem_max[i]) = 0.9;
            location<kernel>(k_stem_max[i]) = tile(X0 , Y0 + 0 +i*2); 
            location<stack >(k_stem_max[i]) = location<kernel>(k_stem_max[i]);
            location<buffer>(k_stem_max[i].out[0]) = location<kernel>(k_stem_max[i]);  

            k_stem[i] = kernel::create_object<layer0_stem>(0);          
            source(k_stem[i]) = "layer0_stem.cpp";
            runtime<ratio>(k_stem[i]) = 0.9;
            location<kernel>(k_stem[i]) = tile(X0 , Y0 + 1 + i*2); 
            location<stack >(k_stem[i]) = location<kernel>(k_stem[i]);
            location<buffer>(k_stem[i].in[0]) = location<kernel>(k_stem[i]);            
            connect<>(k_stem[i].out[0], k_stem_max[i].in[0]);
            
        }

        /////////////////////////////////////////////
        weight_mem = shared_buffer<bfloat16>::create({216}, 1, 3);
        location<buffer>(weight_mem) = location<kernel>(k_stem[0])    + relative_offset({ .col_offset = 0 });  // 只支持列偏移
        num_buffers(weight_mem) = 2;
        for (int i = 0; i < 1; ++i) {
            write_access(weight_mem.in[i]) =
                tiling({
                    .buffer_dimension = {216},   // 整块缓冲  W×H×C
                    .tiling_dimension = {216},              // 一次写 **1 行**
                    .offset           = {0},              // 第 i 个通道起点
                    .tile_traversal   = {
                        { .dimension = 0, .stride = 0, .wrap = 1 },   // W 维：已满 64，不移动
                    }
                });
            connect<>(input_weight[0] , weight_mem.in[i]);
        }
        for (int i = 0; i < 3; ++i) {
            read_access(weight_mem.out[i]) = tiling({
                /* 1. 原始数据区：64 × 64 × 8 */
                .buffer_dimension   = {216},
                .tiling_dimension   = {72},
                .offset             = {72 * i},
                .tile_traversal     = {
                    { .dimension = 0, .stride = 72, .wrap = 1 }  // W 维
                },
            
            });
            connect<>(weight_mem.out[i], k_stem[i].in[0]);
        }

        /////////////////////////////////////////////
        /*
        stem_mem = shared_buffer<bfloat16>::create({64, 64, 1}, 1, 3);
        location<buffer>(stem_mem) = location<kernel>(k_stem[0])    + relative_offset({ .col_offset = 0 });  // 只支持列偏移
        num_buffers(stem_mem) = 2;
        for (int i = 0; i < 1; ++i) {
            write_access(stem_mem.in[i]) =
                tiling({
                    .buffer_dimension = {64, 64, 1},   // 整块缓冲  W×H×C
                    .tiling_dimension = {64,  64, 1},              // 一次写 **1 行**
                    .offset           = {0,  0,  i},              // 第 i 个通道起点
                    .tile_traversal   = {
                        { .dimension = 0, .stride = 0, .wrap = 1 },   // W 维：已满 64，不移动
                        { .dimension = 1, .stride = 0, .wrap = 1 },  // H 维：行号 0→63
                        { .dimension = 2, .stride = 0, .wrap = 1 }    // C 维：一帧 1 通道
                    }
                });
            connect<>(input_image[i] , stem_mem.in[i]);
        }
        for (int i = 0; i < 3; ++i) {
            read_access(stem_mem.out[i]) = tiling({
                // 1. 原始数据区：64 × 64 × 8 
                .buffer_dimension   = {64, 64, 1},
                .tiling_dimension   = {64, 64, 1},
                .offset             = {0, 0,  0},
                .tile_traversal     = {
                    { .dimension = 0, .stride = 0, .wrap = 1 },  // W 维
                    { .dimension = 1, .stride = 0, .wrap = 1 },  // H 维
                    { .dimension = 2, .stride =  0, .wrap = 1 }   // C 维
                },
                .boundary_dimension = {64, 64, 1}
            });
            connect<>(stem_mem.out[i], k_stem[i].in[1]);
        }
        */
       connect<>(input_image[0], k_stem[0].in[1]);
       connect<>(input_image[1], k_stem[1].in[1]);
       connect<>(input_image[2], k_stem[2].in[1]);


        /////////////////////////////////////////////
        stemout_mem = shared_buffer<bfloat16>::create({32, 32, 24}, 3, 1);
        location<buffer>(stemout_mem) = location<kernel>(k_stem[0])    + relative_offset({ .col_offset = 1 });  // 只支持列偏移
        num_buffers(stemout_mem) = 2;
        for (int i = 0; i < 3; ++i) {
            write_access(stemout_mem.in[i]) =
                tiling({
                    .buffer_dimension = {32, 32, 24},   // 整块缓冲  W×H×C
                    .tiling_dimension = {32, 32, 8},              // 一次写 **1 行**
                    .offset           = {0,  0,  8 * i},              // 第 i 个通道起点
                    .tile_traversal   = {
                        { .dimension = 0, .stride = 0, .wrap = 1 },   // W 维：已满 64，不移动
                        { .dimension = 1, .stride = 0, .wrap = 1 },  // H 维：行号 0→63
                        { .dimension = 2, .stride = 0, .wrap = 1 }    // C 维：一帧 1 通道
                    }
                });
            connect<>(k_stem_max[i].out[0] , stemout_mem.in[i]);
        }

        for (int i = 0; i < 1; ++i) {
            read_access(stemout_mem.out[i]) = tiling({
                /* 1. 原始数据区：64 × 64 × 8 */
                .buffer_dimension   = {32, 32, 24},
                .tiling_dimension   = {32, 32, 24},
                .offset             = {0, 0,  0},
                .tile_traversal     = {
                    { .dimension = 0, .stride = 0, .wrap = 1 },  // W 维
                    { .dimension = 1, .stride = 0, .wrap = 1 },  // H 维
                    { .dimension = 2, .stride =  0, .wrap = 1 }   // C 维
                },
            });
            connect<>(stemout_mem.out[i], output[i]);
        }

    }
};




template<
    int  X0  = 16, int  Y0  = 3,            // mem_to_4 起点
    int  X1  = 0, int  Y1  = 0,            // stage-2  起点
    int  NK  = 1                            // stage-2 个数
>
class stem : public adf::graph
{
private:
    static constexpr int NuminKernels = 1; // 用于替换所有的 16
    static constexpr int NumoutKernels = 1; // 用于替换所有的 16
    kernel                   k_stem[3];
    kernel                   k_stem_max[3];
    
public:
    /*** 顶层端口：一进一出 ***/
    adf::port<input>  input_image[1];
    adf::port<input>  input_max[1];
    
    adf::port<input>  input_weight[1];
    adf::port<output> output[1];
    shared_buffer<bfloat16> stem_mem;
    shared_buffer<bfloat16> weight_mem;
    shared_buffer<bfloat16> stemout_mem;


    stem()
    {   
        for (int i = 0; i < 3; ++i) {
            k_stem_max[i] = kernel::create_object<layer0_stem_maxpool>(0);          
            source(k_stem_max[i]) = "layer0_stem_maxpool.cpp";
            runtime<ratio>(k_stem_max[i]) = 0.9;
            location<kernel>(k_stem_max[i]) = tile(X0 , Y0 + 0 +i*2); 
            location<stack >(k_stem_max[i]) = location<kernel>(k_stem_max[i]);
            location<buffer>(k_stem_max[i].out[0]) = location<kernel>(k_stem_max[i]);  

            k_stem[i] = kernel::create_object<layer0_stem>(0);          
            source(k_stem[i]) = "layer0_stem.cpp";
            runtime<ratio>(k_stem[i]) = 0.9;
            location<kernel>(k_stem[i]) = tile(X0 , Y0 + 1 + i*2); 
            location<stack >(k_stem[i]) = location<kernel>(k_stem[i]);
            location<buffer>(k_stem[i].in[0]) = location<kernel>(k_stem[i]);            
            connect<>(k_stem[i].out[0], k_stem_max[i].in[0]);
            
        }

        /////////////////////////////////////////////
        weight_mem = shared_buffer<bfloat16>::create({216}, 1, 3);
        location<buffer>(weight_mem) = location<kernel>(k_stem[0])    + relative_offset({ .col_offset = 0 });  // 只支持列偏移
        num_buffers(weight_mem) = 2;
        for (int i = 0; i < 1; ++i) {
            write_access(weight_mem.in[i]) =
                tiling({
                    .buffer_dimension = {216},   // 整块缓冲  W×H×C
                    .tiling_dimension = {216},              // 一次写 **1 行**
                    .offset           = {0},              // 第 i 个通道起点
                    .tile_traversal   = {
                        { .dimension = 0, .stride = 0, .wrap = 1 },   // W 维：已满 64，不移动
                    }
                });
            connect<>(input_weight[0] , weight_mem.in[i]);
        }
        for (int i = 0; i < 3; ++i) {
            read_access(weight_mem.out[i]) = tiling({
                /* 1. 原始数据区：64 × 64 × 8 */
                .buffer_dimension   = {216},
                .tiling_dimension   = {72},
                .offset             = {72 * i},
                .tile_traversal     = {
                    { .dimension = 0, .stride = 72, .wrap = 1 }  // W 维
                },
            
            });
            connect<>(weight_mem.out[i], k_stem[i].in[0]);
        }

        /////////////////////////////////////////////
        stem_mem = shared_buffer<bfloat16>::create({64, 64, NuminKernels}, NuminKernels, 3);
        location<buffer>(stem_mem) = location<kernel>(k_stem[0])    + relative_offset({ .col_offset = 0 });  // 只支持列偏移
        num_buffers(stem_mem) = 2;
        for (int i = 0; i < NuminKernels; ++i) {
            write_access(stem_mem.in[i]) =
                tiling({
                    .buffer_dimension = {64, 64, NuminKernels},   // 整块缓冲  W×H×C
                    .tiling_dimension = {64,  64, 1},              // 一次写 **1 行**
                    .offset           = {0,  0,  i},              // 第 i 个通道起点
                    .tile_traversal   = {
                        { .dimension = 0, .stride = 0, .wrap = 1 },   // W 维：已满 64，不移动
                        { .dimension = 1, .stride = 0, .wrap = 1 },  // H 维：行号 0→63
                        { .dimension = 2, .stride = 0, .wrap = 1 }    // C 维：一帧 1 通道
                    }
                });
            connect<>(input_image[i] , stem_mem.in[i]);
        }
        for (int i = 0; i < 3; ++i) {
            read_access(stem_mem.out[i]) = tiling({
                /* 1. 原始数据区：64 × 64 × 8 */
                .buffer_dimension   = {64, 64, NuminKernels},
                .tiling_dimension   = {64, 64, 1},
                .offset             = {0, 0,  0},
                .tile_traversal     = {
                    { .dimension = 0, .stride = 0, .wrap = 1 },  // W 维
                    { .dimension = 1, .stride = 0, .wrap = 1 },  // H 维
                    { .dimension = 2, .stride =  0, .wrap = 1 }   // C 维
                },
                .boundary_dimension = {64, 64, NuminKernels}
            });
            connect<>(stem_mem.out[i], k_stem[i].in[1]);
        }


        /////////////////////////////////////////////
        stemout_mem = shared_buffer<bfloat16>::create({32, 32, 24}, 3, 1);
        location<buffer>(stemout_mem) = location<kernel>(k_stem[0])    + relative_offset({ .col_offset = 1 });  // 只支持列偏移
        num_buffers(stemout_mem) = 2;
        for (int i = 0; i < 3; ++i) {
            write_access(stemout_mem.in[i]) =
                tiling({
                    .buffer_dimension = {32, 32, 24},   // 整块缓冲  W×H×C
                    .tiling_dimension = {32, 32, 8},              // 一次写 **1 行**
                    .offset           = {0,  0,  8 * i},              // 第 i 个通道起点
                    .tile_traversal   = {
                        { .dimension = 0, .stride = 0, .wrap = 1 },   // W 维：已满 64，不移动
                        { .dimension = 1, .stride = 0, .wrap = 1 },  // H 维：行号 0→63
                        { .dimension = 2, .stride = 0, .wrap = 1 }    // C 维：一帧 1 通道
                    }
                });
            connect<>(k_stem_max[i].out[0] , stemout_mem.in[i]);
        }

        for (int i = 0; i < 1; ++i) {
            read_access(stemout_mem.out[i]) = tiling({
                /* 1. 原始数据区：64 × 64 × 8 */
                .buffer_dimension   = {32, 32, 24},
                .tiling_dimension   = {32, 32, 24},
                .offset             = {0, 0,  0},
                .tile_traversal     = {
                    { .dimension = 0, .stride = 0, .wrap = 1 },  // W 维
                    { .dimension = 1, .stride = 0, .wrap = 1 },  // H 维
                    { .dimension = 2, .stride =  0, .wrap = 1 }   // C 维
                },
            });
            connect<>(stemout_mem.out[i], output[i]);
        }

    }
};








template<
    int  X0  = 16, int  Y0  = 3,            // mem_to_4 起点
    int  X1  = 0, int  Y1  = 0,            // stage-2  起点
    int  NK  = 1                            // stage-2 个数
>
class layer1_block0_1memout : public adf::graph
{
private:
    kernel                   k_layer1_0_conv1[3];
    kernel                   k_layer1_0_down[3];
    kernel                   k_layer1_0_conv2[3];    
public:
    /*** 顶层端口：一进一出 ***/
    static constexpr int in_C = 24;   // channels / depth
    static constexpr int in_W = 32;   // feature-map width
    static constexpr int in_H = 32;   // feature-map height
    
    static constexpr int out_C = 24;   // channels / depth
    static constexpr int out_W = 32;   // feature-map width
    static constexpr int out_H = 32;   // feature-map height
    
    adf::port<input>  input_image;
    adf::port<input>  input_weight_0_conv1[3];
    //adf::port<input>  input_weight_0_down[3];
    adf::port<input>  input_weight_0_conv2[3];


    adf::port<output> output[1];

    shared_buffer<bfloat16> image_mem0;
    shared_buffer<bfloat16> image_mem1;
    shared_buffer<bfloat16> image_mem2;


    layer1_block0_1memout()
    {   
        //0_conv1
        for (int i = 0; i < 3; ++i) {
            k_layer1_0_conv1[i] = kernel::create_object<layer1_0_conv1>(0);          
            source(k_layer1_0_conv1[i]) = "layer1_0_conv1.cpp";
            runtime<ratio>(k_layer1_0_conv1[i]) = 0.9;
            location<kernel>(k_layer1_0_conv1[i]) = tile(X0 , Y0+3 + i);
            location<stack >(k_layer1_0_conv1[i]) = location<kernel>(k_layer1_0_conv1[i]);
            location<buffer>(k_layer1_0_conv1[i].in[0]) = location<kernel>(k_layer1_0_conv1[i]);            
        }
        //0_down
        for (int i = 0; i < 3; ++i) {
            k_layer1_0_down[i] = kernel::create_object<layer1_0_down>(0);          
            source(k_layer1_0_down[i]) = "layer1_0_down.cpp";
            runtime<ratio>(k_layer1_0_down[i]) = 0.9;
            location<kernel>(k_layer1_0_down[i]) = tile(X0 , Y0 +0+ i);
            location<stack >(k_layer1_0_down[i]) = location<kernel>(k_layer1_0_down[i]);
            //location<buffer>(k_layer1_0_down[i].in[0]) = location<kernel>(k_layer1_0_down[i]);   
            location<buffer>(k_layer1_0_down[i].out[0]) = location<kernel>(k_layer1_0_down[i]);          
        }
        //0_conv2
        for (int i = 0; i < 3; ++i) {
            k_layer1_0_conv2[i] = kernel::create_object<layer1_0_conv2>(0);          
            source(k_layer1_0_conv2[i]) = "layer1_0_conv2.cpp";
            runtime<ratio>(k_layer1_0_conv2[i]) = 0.9;
            location<kernel>(k_layer1_0_conv2[i]) = tile(X0+1 , Y0 +0+ i);
            location<stack >(k_layer1_0_conv2[i]) = location<kernel>(k_layer1_0_conv2[i]);
            location<buffer>(k_layer1_0_conv2[i].in[0]) = location<kernel>(k_layer1_0_conv2[i]);     
        }
        ////////////////////////
        //weights to tiles
        ////////////////////////
        for (int i = 0; i < 3; i++){
        connect<>(input_weight_0_conv1[i], k_layer1_0_conv1[i].in[0]);        
        //connect<>(input_weight_0_down[i], k_layer1_0_down[i].in[0]);
        connect<>(input_weight_0_conv2[i], k_layer1_0_conv2[i].in[0]);        
                        
        }

        ////////////////////////
        //mem0 to (residual and conv1)
        ////////////////////////
        image_mem0 = shared_buffer<bfloat16>::create({in_W, in_H, in_C}, 1, 6);
        location<buffer>(image_mem0) = location<kernel>(k_layer1_0_down[0])    + relative_offset({ .col_offset = -1 });  // 只支持列偏移
        num_buffers(image_mem0) = 2;
        for (int i = 0; i < 1; ++i) {
            write_access(image_mem0.in[i]) =
                tiling({.buffer_dimension = {in_W, in_H, in_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {in_W, in_H, in_C},              // 一次写 **1 行**
                    .offset           = {0, 0,  0},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{.dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(input_image, image_mem0.in[i]);}
        for (int i = 0; i < 6; ++i) {
            read_access(image_mem0.out[i]) 
            = tiling({.buffer_dimension   = {in_W, in_H, in_C},
                .tiling_dimension   = {in_W, in_H, in_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 }, { .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        connect<>(image_mem0.out[0], k_layer1_0_conv1[0].in[1]);
        connect<>(image_mem0.out[1], k_layer1_0_conv1[1].in[1]);
        connect<>(image_mem0.out[2], k_layer1_0_conv1[2].in[1]);
        connect<>(image_mem0.out[3], k_layer1_0_down[0].in[0]);
        connect<>(image_mem0.out[4], k_layer1_0_down[1].in[0]);
        connect<>(image_mem0.out[5], k_layer1_0_down[2].in[0]);

        ////////////////////////
        //0_conv1 to mem1
        ////////////////////////
        image_mem1 = shared_buffer<bfloat16>::create({out_W, out_H, out_C}, 3, 3);
        location<buffer>(image_mem1) = location<kernel>(k_layer1_0_down[0])    + relative_offset({ .col_offset = 0 });  // 只支持列偏移
        num_buffers(image_mem1) = 2;
        for (int i = 0; i < 3; ++i) {
            write_access(image_mem1.in[i]) =
                tiling({.buffer_dimension = {out_W, out_H, out_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {out_W, out_H, out_C/3},              // 一次写 **1 行**
                    .offset           = {0, 0,  (out_C/3)*i},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(k_layer1_0_conv1[i].out[0], image_mem1.in[i]);}
        for (int i = 0; i < 3; ++i) {
            read_access(image_mem1.out[i]) 
            = tiling({.buffer_dimension   = {out_W, out_H, out_C},
                .tiling_dimension   = {out_W, out_H, out_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        ////////////////////////
        //mem1 to 0_conv2
        ////////////////////////            
        connect<>(image_mem1.out[0], k_layer1_0_conv2[0].in[1]);
        connect<>(image_mem1.out[1], k_layer1_0_conv2[1].in[1]);
        connect<>(image_mem1.out[2], k_layer1_0_conv2[2].in[1]);

        ////////////////////////
        //0_residual to 0_conv2
        ////////////////////////
        for (int i = 0; i < 3; ++i) {
            connect<>(k_layer1_0_down[i].out[0], k_layer1_0_conv2[i].in[2]);
        }

        ////////////////////////
        //0_conv2 to mem2
        ////////////////////////
        image_mem2 = shared_buffer<bfloat16>::create({out_W, out_H, out_C}, 3, 1);
        location<buffer>(image_mem2) = location<kernel>(k_layer1_0_down[0])    + relative_offset({ .col_offset = 2 });  // 只支持列偏移
        num_buffers(image_mem2) = 2;
        for (int i = 0; i < 3; ++i) {
            write_access(image_mem2.in[i]) =
                tiling({.buffer_dimension = {out_W, out_H, out_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {out_W, out_H, out_C/3},              // 一次写 **1 行**
                    .offset           = {0, 0,  (out_C/3)*i},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(k_layer1_0_conv2[i].out[0], image_mem2.in[i]);}
        for (int i = 0; i < 1; ++i) {
            read_access(image_mem2.out[i]) 
            = tiling({.buffer_dimension   = {out_W, out_H, out_C},
                .tiling_dimension   = {out_W, out_H, out_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        
        ////////////////////////
        //mem2 to output
        ////////////////////////
        connect<>(image_mem2.out[0], output[0]);
    }
};




template<
    int  X0  = 16, int  Y0  = 3,            // mem_to_4 起点
    int  X1  = 0, int  Y1  = 0,            // stage-2  起点
    int  NK  = 1                            // stage-2 个数
>
class layer2_block0_1memout : public adf::graph
{
private:
    kernel                   k_layer2_0_conv1[3];
    kernel                   k_layer2_0_down[3];
    kernel                   k_layer2_0_conv2[3];    
public:
    /*** 顶层端口：一进一出 ***/
    static constexpr int in_C = 24;   // channels / depth
    static constexpr int in_W = 32;   // feature-map width
    static constexpr int in_H = 32;   // feature-map height
    
    static constexpr int out_C = 48;   // channels / depth
    static constexpr int out_W = 16;   // feature-map width
    static constexpr int out_H = 16;   // feature-map height
    
    adf::port<input>  input_image;
    adf::port<input>  input_weight_0_conv1[3];
    adf::port<input>  input_weight_0_down[3];
    adf::port<input>  input_weight_0_conv2[3];


    adf::port<output> output[1];

    shared_buffer<bfloat16> image_mem0;
    shared_buffer<bfloat16> image_mem1;
    shared_buffer<bfloat16> image_mem2;


    layer2_block0_1memout()
    {   
        //0_conv1
        for (int i = 0; i < 3; ++i) {
            k_layer2_0_conv1[i] = kernel::create_object<layer2_0_conv1>(0);          
            source(k_layer2_0_conv1[i]) = "layer2_0_conv1.cpp";
            runtime<ratio>(k_layer2_0_conv1[i]) = 0.9;
            location<kernel>(k_layer2_0_conv1[i]) = tile(X0 , Y0+3 + i);
            location<stack >(k_layer2_0_conv1[i]) = location<kernel>(k_layer2_0_conv1[i]);
            location<buffer>(k_layer2_0_conv1[i].in[0]) = location<kernel>(k_layer2_0_conv1[i]);            
        }
        //0_down
        for (int i = 0; i < 3; ++i) {
            k_layer2_0_down[i] = kernel::create_object<layer2_0_down>(0);          
            source(k_layer2_0_down[i]) = "layer2_0_down.cpp";
            runtime<ratio>(k_layer2_0_down[i]) = 0.9;
            location<kernel>(k_layer2_0_down[i]) = tile(X0 , Y0 +0+ i);
            location<stack >(k_layer2_0_down[i]) = location<kernel>(k_layer2_0_down[i]);
            location<buffer>(k_layer2_0_down[i].in[0]) = location<kernel>(k_layer2_0_down[i]);   
            location<buffer>(k_layer2_0_down[i].out[0]) = location<kernel>(k_layer2_0_down[i]);          
        }
        //0_conv2
        for (int i = 0; i < 3; ++i) {
            k_layer2_0_conv2[i] = kernel::create_object<layer2_0_conv2>(0);          
            source(k_layer2_0_conv2[i]) = "layer2_0_conv2.cpp";
            runtime<ratio>(k_layer2_0_conv2[i]) = 0.9;
            location<kernel>(k_layer2_0_conv2[i]) = tile(X0+1 , Y0 +0+ i);
            location<stack >(k_layer2_0_conv2[i]) = location<kernel>(k_layer2_0_conv2[i]);
            location<buffer>(k_layer2_0_conv2[i].in[0]) = location<kernel>(k_layer2_0_conv2[i]);     
        }
        ////////////////////////
        //weights to tiles
        ////////////////////////
        for (int i = 0; i < 3; i++){
        connect<>(input_weight_0_conv1[i], k_layer2_0_conv1[i].in[0]);        
        connect<>(input_weight_0_down[i], k_layer2_0_down[i].in[0]);
        connect<>(input_weight_0_conv2[i], k_layer2_0_conv2[i].in[0]);        
                        
        }

        ////////////////////////
        //mem0 to (residual and conv1)
        ////////////////////////
        image_mem0 = shared_buffer<bfloat16>::create({in_W, in_H, in_C}, 1, 6);
        location<buffer>(image_mem0) = location<kernel>(k_layer2_0_down[0])    + relative_offset({ .col_offset = -1 });  // 只支持列偏移
        num_buffers(image_mem0) = 2;
        for (int i = 0; i < 1; ++i) {
            write_access(image_mem0.in[i]) =
                tiling({.buffer_dimension = {in_W, in_H, in_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {in_W, in_H, in_C},              // 一次写 **1 行**
                    .offset           = {0, 0,  0},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{.dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(input_image, image_mem0.in[i]);}
        for (int i = 0; i < 6; ++i) {
            read_access(image_mem0.out[i]) 
            = tiling({.buffer_dimension   = {in_W, in_H, in_C},
                .tiling_dimension   = {in_W, in_H, in_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 }, { .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        connect<>(image_mem0.out[0], k_layer2_0_conv1[0].in[1]);
        connect<>(image_mem0.out[1], k_layer2_0_conv1[1].in[1]);
        connect<>(image_mem0.out[2], k_layer2_0_conv1[2].in[1]);
        connect<>(image_mem0.out[3], k_layer2_0_down[0].in[1]);
        connect<>(image_mem0.out[4], k_layer2_0_down[1].in[1]);
        connect<>(image_mem0.out[5], k_layer2_0_down[2].in[1]);

        ////////////////////////
        //0_conv1 to mem1
        ////////////////////////
        image_mem1 = shared_buffer<bfloat16>::create({out_W, out_H, out_C}, 3, 3);
        location<buffer>(image_mem1) = location<kernel>(k_layer2_0_down[0])    + relative_offset({ .col_offset = 0 });  // 只支持列偏移
        num_buffers(image_mem1) = 2;
        for (int i = 0; i < 3; ++i) {
            write_access(image_mem1.in[i]) =
                tiling({.buffer_dimension = {out_W, out_H, out_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {out_W, out_H, out_C/3},              // 一次写 **1 行**
                    .offset           = {0, 0,  (out_C/3)*i},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(k_layer2_0_conv1[i].out[0], image_mem1.in[i]);}
        for (int i = 0; i < 3; ++i) {
            read_access(image_mem1.out[i]) 
            = tiling({.buffer_dimension   = {out_W, out_H, out_C},
                .tiling_dimension   = {out_W, out_H, out_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        ////////////////////////
        //mem1 to 0_conv2
        ////////////////////////            
        connect<>(image_mem1.out[0], k_layer2_0_conv2[0].in[1]);
        connect<>(image_mem1.out[1], k_layer2_0_conv2[1].in[1]);
        connect<>(image_mem1.out[2], k_layer2_0_conv2[2].in[1]);

        ////////////////////////
        //0_residual to 0_conv2
        ////////////////////////
        for (int i = 0; i < 3; ++i) {
            connect<>(k_layer2_0_down[i].out[0], k_layer2_0_conv2[i].in[2]);
        }

        ////////////////////////
        //0_conv2 to mem2
        ////////////////////////
        image_mem2 = shared_buffer<bfloat16>::create({out_W, out_H, out_C}, 3, 1);
        location<buffer>(image_mem2) = location<kernel>(k_layer2_0_down[0])    + relative_offset({ .col_offset = 2 });  // 只支持列偏移
        num_buffers(image_mem2) = 2;
        for (int i = 0; i < 3; ++i) {
            write_access(image_mem2.in[i]) =
                tiling({.buffer_dimension = {out_W, out_H, out_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {out_W, out_H, out_C/3},              // 一次写 **1 行**
                    .offset           = {0, 0,  (out_C/3)*i},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(k_layer2_0_conv2[i].out[0], image_mem2.in[i]);}
        for (int i = 0; i < 1; ++i) {
            read_access(image_mem2.out[i]) 
            = tiling({.buffer_dimension   = {out_W, out_H, out_C},
                .tiling_dimension   = {out_W, out_H, out_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        
        ////////////////////////
        //mem2 to output
        ////////////////////////
        connect<>(image_mem2.out[0], output[0]);
    }
};




template<
    int  X0  = 16, int  Y0  = 3,            // mem_to_4 起点
    int  X1  = 0, int  Y1  = 0,            // stage-2  起点
    int  NK  = 1                            // stage-2 个数
>
class layer2_block1_1memout : public adf::graph
{
private:
    kernel                   k_layer2_1_conv1[3];
    kernel                   k_layer2_1_down[3];
    kernel                   k_layer2_1_conv2[3];    
public:
    /*** 顶层端口：一进一出 ***/
    static constexpr int in_C = 48;   // channels / depth
    static constexpr int in_W = 16;   // feature-map width
    static constexpr int in_H = 16;   // feature-map height
    
    static constexpr int out_C = 48;   // channels / depth
    static constexpr int out_W = 16;   // feature-map width
    static constexpr int out_H = 16;   // feature-map height
    
    adf::port<input>  input_image;
    adf::port<input>  input_weight_1_conv1[3];
    //adf::port<input>  input_weight_1_down[3];
    adf::port<input>  input_weight_1_conv2[3];


    adf::port<output> output[1];

    shared_buffer<bfloat16> image_mem0;
    shared_buffer<bfloat16> image_mem1;
    shared_buffer<bfloat16> image_mem2;


    layer2_block1_1memout()
    {   
        //0_conv1
        for (int i = 0; i < 3; ++i) {
            k_layer2_1_conv1[i] = kernel::create_object<layer2_1_conv1>(0);          
            source(k_layer2_1_conv1[i]) = "layer2_1_conv1.cpp";
            runtime<ratio>(k_layer2_1_conv1[i]) = 0.9;
            location<kernel>(k_layer2_1_conv1[i]) = tile(X0 , Y0+3 + i);
            location<stack >(k_layer2_1_conv1[i]) = location<kernel>(k_layer2_1_conv1[i]);
            location<buffer>(k_layer2_1_conv1[i].in[0]) = location<kernel>(k_layer2_1_conv1[i]);            
        }
        //0_down
        for (int i = 0; i < 3; ++i) {
            k_layer2_1_down[i] = kernel::create_object<layer2_1_identity>(0);          
            source(k_layer2_1_down[i]) = "layer2_1_identity.cpp";
            runtime<ratio>(k_layer2_1_down[i]) = 0.9;
            location<kernel>(k_layer2_1_down[i]) = tile(X0 , Y0 +0+ i);
            location<stack >(k_layer2_1_down[i]) = location<kernel>(k_layer2_1_down[i]);
            //location<buffer>(k_layer2_1_down[i].in[0]) = location<kernel>(k_layer2_1_down[i]);   
            location<buffer>(k_layer2_1_down[i].out[0]) = location<kernel>(k_layer2_1_down[i]);          
        }
        //0_conv2
        for (int i = 0; i < 3; ++i) {
            k_layer2_1_conv2[i] = kernel::create_object<layer2_1_conv2>(0);          
            source(k_layer2_1_conv2[i]) = "layer2_1_conv2.cpp";
            runtime<ratio>(k_layer2_1_conv2[i]) = 0.9;
            location<kernel>(k_layer2_1_conv2[i]) = tile(X0+1 , Y0 +0+ i);
            location<stack >(k_layer2_1_conv2[i]) = location<kernel>(k_layer2_1_conv2[i]);
            location<buffer>(k_layer2_1_conv2[i].in[0]) = location<kernel>(k_layer2_1_conv2[i]);     
        }
        ////////////////////////
        //weights to tiles
        ////////////////////////
        for (int i = 0; i < 3; i++){
        connect<>(input_weight_1_conv1[i], k_layer2_1_conv1[i].in[0]);        
        //connect<>(input_weight_1_down[i], k_layer2_1_down[i].in[0]);
        connect<>(input_weight_1_conv2[i], k_layer2_1_conv2[i].in[0]);        
                        
        }

        ////////////////////////
        //mem0 to (residual and conv1)
        ////////////////////////
        image_mem0 = shared_buffer<bfloat16>::create({in_W, in_H, in_C}, 1, 6);
        location<buffer>(image_mem0) = location<kernel>(k_layer2_1_down[0])    + relative_offset({ .col_offset = -1 });  // 只支持列偏移
        num_buffers(image_mem0) = 2;
        for (int i = 0; i < 1; ++i) {
            write_access(image_mem0.in[i]) =
                tiling({.buffer_dimension = {in_W, in_H, in_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {in_W, in_H, in_C},              // 一次写 **1 行**
                    .offset           = {0, 0,  0},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{.dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(input_image, image_mem0.in[i]);}
        for (int i = 0; i < 3; ++i) {
            read_access(image_mem0.out[i]) 
            = tiling({.buffer_dimension   = {in_W, in_H, in_C},
                .tiling_dimension   = {in_W, in_H, in_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 }, { .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        for (int i = 0; i < 3; ++i) {
            read_access(image_mem0.out[i+3]) 
            = tiling({.buffer_dimension   = {in_W, in_H, in_C},
                .tiling_dimension   = {in_W, in_H, in_C/3},
                .offset             = {0, 0,  (in_C/3)*i},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 }, { .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}

        connect<>(image_mem0.out[0], k_layer2_1_conv1[0].in[1]);
        connect<>(image_mem0.out[1], k_layer2_1_conv1[1].in[1]);
        connect<>(image_mem0.out[2], k_layer2_1_conv1[2].in[1]);
        connect<>(image_mem0.out[3], k_layer2_1_down[0].in[0]);
        connect<>(image_mem0.out[4], k_layer2_1_down[1].in[0]);
        connect<>(image_mem0.out[5], k_layer2_1_down[2].in[0]);

        ////////////////////////
        //0_conv1 to mem1
        ////////////////////////
        image_mem1 = shared_buffer<bfloat16>::create({out_W, out_H, out_C}, 3, 3);
        location<buffer>(image_mem1) = location<kernel>(k_layer2_1_down[0])    + relative_offset({ .col_offset = 0 });  // 只支持列偏移
        num_buffers(image_mem1) = 2;
        for (int i = 0; i < 3; ++i) {
            write_access(image_mem1.in[i]) =
                tiling({.buffer_dimension = {out_W, out_H, out_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {out_W, out_H, out_C/3},              // 一次写 **1 行**
                    .offset           = {0, 0,  (out_C/3)*i},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(k_layer2_1_conv1[i].out[0], image_mem1.in[i]);}
        for (int i = 0; i < 3; ++i) {
            read_access(image_mem1.out[i]) 
            = tiling({.buffer_dimension   = {out_W, out_H, out_C},
                .tiling_dimension   = {out_W, out_H, out_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        ////////////////////////
        //mem1 to 0_conv2
        ////////////////////////            
        connect<>(image_mem1.out[0], k_layer2_1_conv2[0].in[1]);
        connect<>(image_mem1.out[1], k_layer2_1_conv2[1].in[1]);
        connect<>(image_mem1.out[2], k_layer2_1_conv2[2].in[1]);

        ////////////////////////
        //0_residual to 0_conv2
        ////////////////////////
        for (int i = 0; i < 3; ++i) {
            connect<>(k_layer2_1_down[i].out[0], k_layer2_1_conv2[i].in[2]);
        }

        ////////////////////////
        //0_conv2 to mem2
        ////////////////////////
        image_mem2 = shared_buffer<bfloat16>::create({out_W, out_H, out_C}, 3, 1);
        location<buffer>(image_mem2) = location<kernel>(k_layer2_1_down[0])    + relative_offset({ .col_offset = 2 });  // 只支持列偏移
        num_buffers(image_mem2) = 2;
        for (int i = 0; i < 3; ++i) {
            write_access(image_mem2.in[i]) =
                tiling({.buffer_dimension = {out_W, out_H, out_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {out_W, out_H, out_C/3},              // 一次写 **1 行**
                    .offset           = {0, 0,  (out_C/3)*i},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(k_layer2_1_conv2[i].out[0], image_mem2.in[i]);}
        for (int i = 0; i < 1; ++i) {
            read_access(image_mem2.out[i]) 
            = tiling({.buffer_dimension   = {out_W, out_H, out_C},
                .tiling_dimension   = {out_W, out_H, out_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        
        ////////////////////////
        //mem2 to output
        ////////////////////////
        connect<>(image_mem2.out[0], output[0]);
    }
};



















template<
    int  X0  = 16, int  Y0  = 3,            // mem_to_4 起点
    int  X1  = 0, int  Y1  = 0,            // stage-2  起点
    int  NK  = 1                            // stage-2 个数
>
class layer3_block0_1memout : public adf::graph
{
private:
    kernel                   k_layer3_0_conv1[3];
    kernel                   k_layer3_0_down[3];
    kernel                   k_layer3_0_conv2[3];    
public:
    /*** 顶层端口：一进一出 ***/
    static constexpr int in_C = 48;   // channels / depth
    static constexpr int in_W = 16;   // feature-map width
    static constexpr int in_H = 16;   // feature-map height
    
    static constexpr int out_C = 72;   // channels / depth
    static constexpr int out_W = 8;   // feature-map width
    static constexpr int out_H = 8;   // feature-map height
    
    adf::port<input>  input_image;
    adf::port<input>  input_weight_0_conv1[3];
    adf::port<input>  input_weight_0_down[3];
    adf::port<input>  input_weight_0_conv2[3];


    adf::port<output> output[1];

    shared_buffer<bfloat16> image_mem0;
    shared_buffer<bfloat16> image_mem1;
    shared_buffer<bfloat16> image_mem2;


    layer3_block0_1memout()
    {   
        //0_conv1
        for (int i = 0; i < 3; ++i) {
            k_layer3_0_conv1[i] = kernel::create_object<layer3_0_conv1>(0);          
            source(k_layer3_0_conv1[i]) = "layer3_0_conv1.cpp";
            runtime<ratio>(k_layer3_0_conv1[i]) = 0.9;
            location<kernel>(k_layer3_0_conv1[i]) = tile(X0 , Y0+3 + i);
            location<stack >(k_layer3_0_conv1[i]) = location<kernel>(k_layer3_0_conv1[i]);
            location<buffer>(k_layer3_0_conv1[i].in[0]) = location<kernel>(k_layer3_0_conv1[i]);            
        }
        //0_down
        for (int i = 0; i < 3; ++i) {
            k_layer3_0_down[i] = kernel::create_object<layer3_0_down>(0);          
            source(k_layer3_0_down[i]) = "layer3_0_down.cpp";
            runtime<ratio>(k_layer3_0_down[i]) = 0.9;
            location<kernel>(k_layer3_0_down[i]) = tile(X0 , Y0 +0+ i);
            location<stack >(k_layer3_0_down[i]) = location<kernel>(k_layer3_0_down[i]);
            location<buffer>(k_layer3_0_down[i].in[0]) = location<kernel>(k_layer3_0_down[i]);
            //location<buffer>(k_layer3_0_down[i].out[0]) = location<kernel>(k_layer3_0_down[i]);          
        }
        //0_conv2
        for (int i = 0; i < 3; ++i) {
            k_layer3_0_conv2[i] = kernel::create_object<layer3_0_conv2>(0);          
            source(k_layer3_0_conv2[i]) = "layer3_0_conv2.cpp";
            runtime<ratio>(k_layer3_0_conv2[i]) = 0.9;
            location<kernel>(k_layer3_0_conv2[i]) = tile(X0+1 , Y0 +0+ 2*i);
            location<stack >(k_layer3_0_conv2[i]) = bank(X0 + 1,  Y0 +1+ 2*i, 0);           
            location<buffer>(k_layer3_0_conv2[i].in[0]) = location<kernel>(k_layer3_0_conv2[i]);    
          
        }
        ////////////////////////
        //weights to tiles
        ////////////////////////
        for (int i = 0; i < 3; i++){
        connect<>(input_weight_0_conv1[i], k_layer3_0_conv1[i].in[0]);        
        connect<>(input_weight_0_down[i], k_layer3_0_down[i].in[0]);
        connect<>(input_weight_0_conv2[i], k_layer3_0_conv2[i].in[0]);        
                        
        }

        ////////////////////////
        //mem0 to (residual and conv1)
        ////////////////////////
        image_mem0 = shared_buffer<bfloat16>::create({in_W, in_H, in_C}, 1, 6);
        location<buffer>(image_mem0) = location<kernel>(k_layer3_0_down[0])    + relative_offset({ .col_offset = -1 });  // 只支持列偏移
        num_buffers(image_mem0) = 2;
        for (int i = 0; i < 1; ++i) {
            write_access(image_mem0.in[i]) =
                tiling({.buffer_dimension = {in_W, in_H, in_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {in_W, in_H, in_C},              // 一次写 **1 行**
                    .offset           = {0, 0,  0},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{.dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(input_image, image_mem0.in[i]);}
        for (int i = 0; i < 6; ++i) {
            read_access(image_mem0.out[i]) 
            = tiling({.buffer_dimension   = {in_W, in_H, in_C},
                .tiling_dimension   = {in_W, in_H, in_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 }, { .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        connect<>(image_mem0.out[0], k_layer3_0_conv1[0].in[1]);
        connect<>(image_mem0.out[1], k_layer3_0_conv1[1].in[1]);
        connect<>(image_mem0.out[2], k_layer3_0_conv1[2].in[1]);
        connect<>(image_mem0.out[3], k_layer3_0_down[0].in[1]);
        connect<>(image_mem0.out[4], k_layer3_0_down[1].in[1]);
        connect<>(image_mem0.out[5], k_layer3_0_down[2].in[1]);

        ////////////////////////
        //0_conv1 to mem1
        ////////////////////////
        image_mem1 = shared_buffer<bfloat16>::create({out_W, out_H, out_C}, 3, 3);
        location<buffer>(image_mem1) = location<kernel>(k_layer3_0_down[0])    + relative_offset({ .col_offset = 0 });  // 只支持列偏移
        num_buffers(image_mem1) = 2;
        for (int i = 0; i < 3; ++i) {
            write_access(image_mem1.in[i]) =
                tiling({.buffer_dimension = {out_W, out_H, out_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {out_W, out_H, out_C/3},              // 一次写 **1 行**
                    .offset           = {0, 0,  (out_C/3)*i},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(k_layer3_0_conv1[i].out[0], image_mem1.in[i]);}
        for (int i = 0; i < 3; ++i) {
            read_access(image_mem1.out[i]) 
            = tiling({.buffer_dimension   = {out_W, out_H, out_C},
                .tiling_dimension   = {out_W, out_H, out_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        ////////////////////////
        //mem1 to 0_conv2
        ////////////////////////            
        connect<>(image_mem1.out[0], k_layer3_0_conv2[0].in[1]);
        connect<>(image_mem1.out[1], k_layer3_0_conv2[1].in[1]);
        connect<>(image_mem1.out[2], k_layer3_0_conv2[2].in[1]);

        ////////////////////////
        //0_residual to 0_conv2
        ////////////////////////
        for (int i = 0; i < 3; ++i) {
            connect<>(k_layer3_0_down[i].out[0], k_layer3_0_conv2[i].in[2]);
        }

        ////////////////////////
        //0_conv2 to mem2
        ////////////////////////
        image_mem2 = shared_buffer<bfloat16>::create({out_W, out_H, out_C}, 3, 1);
        location<buffer>(image_mem2) = location<kernel>(k_layer3_0_down[0])    + relative_offset({ .col_offset = 2 });  // 只支持列偏移
        num_buffers(image_mem2) = 2;
        for (int i = 0; i < 3; ++i) {
            write_access(image_mem2.in[i]) =
                tiling({.buffer_dimension = {out_W, out_H, out_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {out_W, out_H, out_C/3},              // 一次写 **1 行**
                    .offset           = {0, 0,  (out_C/3)*i},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(k_layer3_0_conv2[i].out[0], image_mem2.in[i]);}
        for (int i = 0; i < 1; ++i) {
            read_access(image_mem2.out[i]) 
            = tiling({.buffer_dimension   = {out_W, out_H, out_C},
                .tiling_dimension   = {out_W, out_H, out_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        
        ////////////////////////
        //mem2 to output
        ////////////////////////
        connect<>(image_mem2.out[0], output[0]);
    }
};



template<
    int  X0  = 16, int  Y0  = 3,            // mem_to_4 起点
    int  X1  = 0, int  Y1  = 0,            // stage-2  起点
    int  NK  = 1                            // stage-2 个数
>
class layer3_block1_1memout : public adf::graph
{
private:
    kernel                   k_layer3_0_conv1[3];
    kernel                   k_layer3_0_down[3];
    kernel                   k_layer3_0_conv2[3];    
public:
    /*** 顶层端口：一进一出 ***/
    static constexpr int in_C = 72;   // channels / depth
    static constexpr int in_W = 8;   // feature-map width
    static constexpr int in_H = 8;   // feature-map height
    
    static constexpr int out_C = 72;   // channels / depth
    static constexpr int out_W = 8;   // feature-map width
    static constexpr int out_H = 8;   // feature-map height
    
    adf::port<input>  input_image;
    adf::port<input>  input_weight_0_conv1[3];
    //adf::port<input>  input_weight_0_down[3];
    adf::port<input>  input_weight_0_conv2[3];


    adf::port<output> output[1];

    shared_buffer<bfloat16> image_mem0;
    shared_buffer<bfloat16> image_mem1;
    shared_buffer<bfloat16> image_mem2;


    layer3_block1_1memout()
    {   
        //0_conv1
        for (int i = 0; i < 3; ++i) {
            k_layer3_0_conv1[i] = kernel::create_object<layer3_1_conv1>(0);          
            source(k_layer3_0_conv1[i]) = "layer3_1_conv1.cpp";
            runtime<ratio>(k_layer3_0_conv1[i]) = 0.9;
            location<kernel>(k_layer3_0_conv1[i]) = tile(X0 , Y0+3 + i);
            //location<stack >(k_layer3_0_conv1[i]) = location<kernel>(k_layer3_0_conv1[i]);
            //location<buffer>(k_layer3_0_conv1[i].in[0]) = location<kernel>(k_layer3_0_conv1[i]);            
        }
        //0_down
        for (int i = 0; i < 3; ++i) {
            k_layer3_0_down[i] = kernel::create_object<layer3_1_identity>(0);          
            source(k_layer3_0_down[i]) = "layer3_1_identity.cpp";
            runtime<ratio>(k_layer3_0_down[i]) = 0.9;
            location<kernel>(k_layer3_0_down[i]) = tile(X0 , Y0 +0+ i);
            //location<stack >(k_layer3_0_down[i]) = location<kernel>(k_layer3_0_down[i]);
            //location<buffer>(k_layer3_0_down[i].in[0]) = location<kernel>(k_layer3_0_down[i]);
            //location<buffer>(k_layer3_0_down[i].out[0]) = location<kernel>(k_layer3_0_down[i]);          
        }
        //0_conv2
        for (int i = 0; i < 3; ++i) {
            k_layer3_0_conv2[i] = kernel::create_object<layer3_1_conv2>(0);          
            source(k_layer3_0_conv2[i]) = "layer3_1_conv2.cpp";
            runtime<ratio>(k_layer3_0_conv2[i]) = 0.9;
            location<kernel>(k_layer3_0_conv2[i]) = tile(X0+1 , Y0 +0+ 2*i);
            //location<stack >(k_layer3_0_conv2[i]) = bank(X0 + 1,  Y0 +1+ 2*i, 0);           
            //location<buffer>(k_layer3_0_conv2[i].in[0]) = location<kernel>(k_layer3_0_conv2[i]);    
          
        }
        ////////////////////////
        //weights to tiles
        ////////////////////////
        for (int i = 0; i < 3; i++){
        connect<>(input_weight_0_conv1[i], k_layer3_0_conv1[i].in[0]);        
        //connect<>(input_weight_0_down[i], k_layer3_0_down[i].in[0]);
        connect<>(input_weight_0_conv2[i], k_layer3_0_conv2[i].in[0]);        
                        
        }

        ////////////////////////
        //mem0 to (residual and conv1)
        ////////////////////////
        image_mem0 = shared_buffer<bfloat16>::create({in_W, in_H, in_C}, 1, 6);
        location<buffer>(image_mem0) = location<kernel>(k_layer3_0_down[0])    + relative_offset({ .col_offset = -1 });  // 只支持列偏移
        num_buffers(image_mem0) = 2;
        for (int i = 0; i < 1; ++i) {
            write_access(image_mem0.in[i]) =
                tiling({.buffer_dimension = {in_W, in_H, in_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {in_W, in_H, in_C},              // 一次写 **1 行**
                    .offset           = {0, 0,  0},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{.dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(input_image, image_mem0.in[i]);}
        for (int i = 0; i < 3; ++i) {
            read_access(image_mem0.out[i]) 
            = tiling({.buffer_dimension   = {in_W, in_H, in_C},
                .tiling_dimension   = {in_W, in_H, in_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 }, { .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        for (int i = 0; i < 3; ++i) {
            read_access(image_mem0.out[i+3]) 
            = tiling({.buffer_dimension   = {in_W, in_H, in_C},
                .tiling_dimension   = {in_W, in_H, in_C/3},
                .offset             = {0, 0,  (in_C/3)*i},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 }, { .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
    
        connect<>(image_mem0.out[0], k_layer3_0_conv1[0].in[1]);
        connect<>(image_mem0.out[1], k_layer3_0_conv1[1].in[1]);
        connect<>(image_mem0.out[2], k_layer3_0_conv1[2].in[1]);
        connect<>(image_mem0.out[3], k_layer3_0_down[0].in[0]);
        connect<>(image_mem0.out[4], k_layer3_0_down[1].in[0]);
        connect<>(image_mem0.out[5], k_layer3_0_down[2].in[0]);

        ////////////////////////
        //0_conv1 to mem1
        ////////////////////////
        image_mem1 = shared_buffer<bfloat16>::create({out_W, out_H, out_C}, 3, 3);
        location<buffer>(image_mem1) = location<kernel>(k_layer3_0_down[0])    + relative_offset({ .col_offset = 0 });  // 只支持列偏移
        num_buffers(image_mem1) = 2;
        for (int i = 0; i < 3; ++i) {
            write_access(image_mem1.in[i]) =
                tiling({.buffer_dimension = {out_W, out_H, out_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {out_W, out_H, out_C/3},              // 一次写 **1 行**
                    .offset           = {0, 0,  (out_C/3)*i},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(k_layer3_0_conv1[i].out[0], image_mem1.in[i]);}
        for (int i = 0; i < 3; ++i) {
            read_access(image_mem1.out[i]) 
            = tiling({.buffer_dimension   = {out_W, out_H, out_C},
                .tiling_dimension   = {out_W, out_H, out_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        ////////////////////////
        //mem1 to 0_conv2
        ////////////////////////            
        connect<>(image_mem1.out[0], k_layer3_0_conv2[0].in[1]);
        connect<>(image_mem1.out[1], k_layer3_0_conv2[1].in[1]);
        connect<>(image_mem1.out[2], k_layer3_0_conv2[2].in[1]);

        ////////////////////////
        //0_residual to 0_conv2
        ////////////////////////
        for (int i = 0; i < 3; ++i) {
            connect<>(k_layer3_0_down[i].out[0], k_layer3_0_conv2[i].in[2]);
        }

        ////////////////////////
        //0_conv2 to mem2
        ////////////////////////
        image_mem2 = shared_buffer<bfloat16>::create({out_W, out_H, out_C}, 3, 1);
        location<buffer>(image_mem2) = location<kernel>(k_layer3_0_down[0])    + relative_offset({ .col_offset = 2 });  // 只支持列偏移
        num_buffers(image_mem2) = 2;
        for (int i = 0; i < 3; ++i) {
            write_access(image_mem2.in[i]) =
                tiling({.buffer_dimension = {out_W, out_H, out_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {out_W, out_H, out_C/3},              // 一次写 **1 行**
                    .offset           = {0, 0,  (out_C/3)*i},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(k_layer3_0_conv2[i].out[0], image_mem2.in[i]);}
        for (int i = 0; i < 1; ++i) {
            read_access(image_mem2.out[i]) 
            = tiling({.buffer_dimension   = {out_W, out_H, out_C},
                .tiling_dimension   = {out_W, out_H, out_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        
        ////////////////////////
        //mem2 to output
        ////////////////////////
        connect<>(image_mem2.out[0], output[0]);
    }
};






template<
    int  X0  = 16, int  Y0  = 3,            // mem_to_4 起点
    int  X1  = 0, int  Y1  = 0,            // stage-2  起点
    int  NK  = 1                            // stage-2 个数
>
class layer4_block0_1memout : public adf::graph
{
private:
    kernel                   k_layer4_0_conv1[3];
    kernel                   k_layer4_0_down[3];
    kernel                   k_layer4_0_conv2[3];    
public:
    /*** 顶层端口：一进一出 ***/
    static constexpr int in_C = 72;   // channels / depth
    static constexpr int in_W = 8;   // feature-map width
    static constexpr int in_H = 8;   // feature-map height
    
    static constexpr int out_C = 96;   // channels / depth
    static constexpr int out_W = 4;   // feature-map width
    static constexpr int out_H = 4;   // feature-map height
    
    adf::port<input>  input_image;
    adf::port<input>  input_weight_0_conv1[3];
    adf::port<input>  input_weight_0_conv2[3];
    adf::port<input>  input_weight_1_conv1[3];
    adf::port<input>  input_weight_1_conv2[3];
    adf::port<input>  input_weight_0_down[3];


    adf::port<output> output[1];

    shared_buffer<bfloat16> image_mem0;
    shared_buffer<bfloat16> image_mem1;
    shared_buffer<bfloat16> image_mem2;


    layer4_block0_1memout()
    {   
        //0_conv1
        for (int i = 0; i < 3; ++i) {
            k_layer4_0_conv1[i] = kernel::create_object<layer4_0_conv1>(0);          
            source(k_layer4_0_conv1[i]) = "layer4_0_conv1.cpp";
            runtime<ratio>(k_layer4_0_conv1[i]) = 0.9;
            location<kernel>(k_layer4_0_conv1[i]) = tile(X0 , Y0+3 + 2*i);
            //location<stack >(k_layer4_0_conv1[i]) = location<kernel>(k_layer4_0_conv1[i]);
            //location<buffer>(k_layer4_0_conv1[i].in[0]) = location<kernel>(k_layer4_0_conv1[i]);            
        }
        //0_down
        for (int i = 0; i < 3; ++i) {
            k_layer4_0_down[i] = kernel::create_object<layer4_0_down>(0);          
            source(k_layer4_0_down[i]) = "layer4_0_down.cpp";
            runtime<ratio>(k_layer4_0_down[i]) = 0.9;
            location<kernel>(k_layer4_0_down[i]) = tile(X0 , Y0 +0+ i);
            //location<stack >(k_layer4_0_down[i]) = location<kernel>(k_layer4_0_down[i]);
            //location<buffer>(k_layer4_0_down[i].in[0]) = location<kernel>(k_layer4_0_down[i]);
            //location<buffer>(k_layer4_0_down[i].out[0]) = location<kernel>(k_layer4_0_down[i]);          
        }
        //0_conv2
        for (int i = 0; i < 3; ++i) {
            k_layer4_0_conv2[i] = kernel::create_object<layer4_0_conv2>(0);          
            source(k_layer4_0_conv2[i]) = "layer4_0_conv2.cpp";
            runtime<ratio>(k_layer4_0_conv2[i]) = 0.9;
            //location<kernel>(k_layer4_0_conv2[i]) = tile(X0+2 , Y0 +0+ 2*i);
            //location<stack >(k_layer4_0_conv2[i]) = bank(X0 + 1,  Y0 +1+ 2*i, 0);           
            //location<buffer>(k_layer4_0_conv2[i].in[0]) = location<kernel>(k_layer4_0_conv2[i]);              
        }
        location<kernel>(k_layer4_0_conv2[0]) = tile(X0+1 , Y0 +1);
        location<kernel>(k_layer4_0_conv2[1]) = tile(X0+1 , Y0 +3);
        location<kernel>(k_layer4_0_conv2[2]) = tile(X0+1 , Y0 +6);
        ////////////////////////
        //weights to tiles
        ////////////////////////
        for (int i = 0; i < 3; i++){
        connect<>(input_weight_0_conv1[i], k_layer4_0_conv1[i].in[0]);        
        connect<>(input_weight_1_conv1[i], k_layer4_0_conv1[i].in[1]);
        connect<>(input_weight_0_conv2[i], k_layer4_0_conv2[i].in[0]);
        connect<>(input_weight_1_conv2[i], k_layer4_0_conv2[i].in[1]);
        
        connect<>(input_weight_0_down[i], k_layer4_0_down[i].in[0]);
                        
        }

        ////////////////////////
        //mem0 to (residual and conv1)
        ////////////////////////
        image_mem0 = shared_buffer<bfloat16>::create({in_W, in_H, in_C}, 1, 6);
        location<buffer>(image_mem0) = location<kernel>(k_layer4_0_down[0])    + relative_offset({ .col_offset = -1 });  // 只支持列偏移
        num_buffers(image_mem0) = 2;
        for (int i = 0; i < 1; ++i) {
            write_access(image_mem0.in[i]) =
                tiling({.buffer_dimension = {in_W, in_H, in_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {in_W, in_H, in_C},              // 一次写 **1 行**
                    .offset           = {0, 0,  0},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{.dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(input_image, image_mem0.in[i]);}
        for (int i = 0; i < 6; ++i) {
            read_access(image_mem0.out[i]) 
            = tiling({.buffer_dimension   = {in_W, in_H, in_C},
                .tiling_dimension   = {in_W, in_H, in_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 }, { .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        connect<>(image_mem0.out[0], k_layer4_0_conv1[0].in[2]);
        connect<>(image_mem0.out[1], k_layer4_0_conv1[1].in[2]);
        connect<>(image_mem0.out[2], k_layer4_0_conv1[2].in[2]);
        connect<>(image_mem0.out[3], k_layer4_0_down[0].in[1]);
        connect<>(image_mem0.out[4], k_layer4_0_down[1].in[1]);
        connect<>(image_mem0.out[5], k_layer4_0_down[2].in[1]);

        ////////////////////////
        //0_conv1 to mem1
        ////////////////////////
        image_mem1 = shared_buffer<bfloat16>::create({out_W, out_H, out_C}, 3, 3);
        location<buffer>(image_mem1) = location<kernel>(k_layer4_0_down[0])    + relative_offset({ .col_offset = 0 });  // 只支持列偏移
        num_buffers(image_mem1) = 2;
        for (int i = 0; i < 3; ++i) {
            write_access(image_mem1.in[i]) =
                tiling({.buffer_dimension = {out_W, out_H, out_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {out_W, out_H, out_C/3},              // 一次写 **1 行**
                    .offset           = {0, 0,  (out_C/3)*i},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(k_layer4_0_conv1[i].out[0], image_mem1.in[i]);}
        for (int i = 0; i < 3; ++i) {
            read_access(image_mem1.out[i]) 
            = tiling({.buffer_dimension   = {out_W, out_H, out_C},
                .tiling_dimension   = {out_W, out_H, out_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        ////////////////////////
        //mem1 to 0_conv2
        ////////////////////////            
        connect<>(image_mem1.out[0], k_layer4_0_conv2[0].in[2]);
        connect<>(image_mem1.out[1], k_layer4_0_conv2[1].in[2]);
        connect<>(image_mem1.out[2], k_layer4_0_conv2[2].in[2]);

        ////////////////////////
        //0_residual to 0_conv2
        ////////////////////////
        for (int i = 0; i < 3; ++i) {
            connect<>(k_layer4_0_down[i].out[0], k_layer4_0_conv2[i].in[3]);
        }

        ////////////////////////
        //0_conv2 to mem2
        ////////////////////////
        image_mem2 = shared_buffer<bfloat16>::create({out_W, out_H, out_C}, 3, 1);
        location<buffer>(image_mem2) = location<kernel>(k_layer4_0_down[0])    + relative_offset({ .col_offset = 2 });  // 只支持列偏移
        num_buffers(image_mem2) = 2;
        for (int i = 0; i < 3; ++i) {
            write_access(image_mem2.in[i]) =
                tiling({.buffer_dimension = {out_W, out_H, out_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {out_W, out_H, out_C/3},              // 一次写 **1 行**
                    .offset           = {0, 0,  (out_C/3)*i},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(k_layer4_0_conv2[i].out[0], image_mem2.in[i]);}
        for (int i = 0; i < 1; ++i) {
            read_access(image_mem2.out[i]) 
            = tiling({.buffer_dimension   = {out_W, out_H, out_C},
                .tiling_dimension   = {out_W, out_H, out_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        
        ////////////////////////
        //mem2 to output
        ////////////////////////
        connect<>(image_mem2.out[0], output[0]);
    }
};



template<
    int  X0  = 16, int  Y0  = 3,            // mem_to_4 起点
    int  X1  = 0, int  Y1  = 0,            // stage-2  起点
    int  NK  = 1                            // stage-2 个数
>
class layer4_block1_1memout : public adf::graph
{
private:
    kernel                   k_layer4_0_conv1[3];
    kernel                   k_layer4_0_down[3];
    kernel                   k_layer4_0_conv2[3];    
public:
    /*** 顶层端口：一进一出 ***/
    static constexpr int in_C = 96;   // channels / depth
    static constexpr int in_W = 4;   // feature-map width
    static constexpr int in_H = 4;   // feature-map height
    
    static constexpr int out_C = 96;   // channels / depth
    static constexpr int out_W = 4;   // feature-map width
    static constexpr int out_H = 4;   // feature-map height
    
    adf::port<input>  input_image;
    adf::port<input>  input_weight_0_conv1[3];
    adf::port<input>  input_weight_1_conv1[3];
    //adf::port<input>  input_weight_0_down[3];
    adf::port<input>  input_weight_0_conv2[3];
    adf::port<input>  input_weight_1_conv2[3];


    adf::port<output> output[1];

    shared_buffer<bfloat16> image_mem0;
    shared_buffer<bfloat16> image_mem1;
    shared_buffer<bfloat16> image_mem2;


    layer4_block1_1memout()
    {   
        //0_conv1
        for (int i = 0; i < 3; ++i) {
            k_layer4_0_conv1[i] = kernel::create_object<layer4_1_conv1>(0);          
            source(k_layer4_0_conv1[i]) = "layer4_1_conv1.cpp";
            runtime<ratio>(k_layer4_0_conv1[i]) = 0.9;
            location<kernel>(k_layer4_0_conv1[i]) = tile(X0 , Y0+3 + 2*i);
            //location<stack >(k_layer4_0_conv1[i]) = location<kernel>(k_layer4_0_conv1[i]);
            //location<buffer>(k_layer4_0_conv1[i].in[0]) = location<kernel>(k_layer4_0_conv1[i]);            
        }
        //0_down
        for (int i = 0; i < 3; ++i) {
            k_layer4_0_down[i] = kernel::create_object<layer4_1_identity>(0);          
            source(k_layer4_0_down[i]) = "layer4_1_identity.cpp";
            runtime<ratio>(k_layer4_0_down[i]) = 0.9;
            location<kernel>(k_layer4_0_down[i]) = tile(X0 , Y0 +0+ i);
            //location<stack >(k_layer4_0_down[i]) = location<kernel>(k_layer4_0_down[i]);
            //location<buffer>(k_layer4_0_down[i].in[0]) = location<kernel>(k_layer4_0_down[i]);
            //location<buffer>(k_layer4_0_down[i].out[0]) = location<kernel>(k_layer4_0_down[i]);          
        }
        //0_conv2
        for (int i = 0; i < 3; ++i) {
            k_layer4_0_conv2[i] = kernel::create_object<layer4_1_conv2>(0);          
            source(k_layer4_0_conv2[i]) = "layer4_1_conv2.cpp";
            runtime<ratio>(k_layer4_0_conv2[i]) = 0.9;
            //location<kernel>(k_layer4_0_conv2[i]) = tile(X0+2 , Y0 +0+ 2*i);
            //location<stack >(k_layer4_0_conv2[i]) = bank(X0 + 1,  Y0 +1+ 2*i, 0);           
            //location<buffer>(k_layer4_0_conv2[i].in[0]) = location<kernel>(k_layer4_0_conv2[i]);    
          
        }
        location<kernel>(k_layer4_0_conv2[0]) = tile(X0+1 , Y0 +1);
        location<kernel>(k_layer4_0_conv2[1]) = tile(X0+1 , Y0 +3);
        location<kernel>(k_layer4_0_conv2[2]) = tile(X0+1 , Y0 +6);
        ////////////////////////
        //weights to tiles
        ////////////////////////
        for (int i = 0; i < 3; i++){
        connect<>(input_weight_0_conv1[i], k_layer4_0_conv1[i].in[0]);        
        connect<>(input_weight_1_conv1[i], k_layer4_0_conv1[i].in[1]);
        //connect<>(input_weight_0_down[i], k_layer4_0_down[i].in[0]);
        connect<>(input_weight_0_conv2[i], k_layer4_0_conv2[i].in[0]);        
        connect<>(input_weight_1_conv2[i], k_layer4_0_conv2[i].in[1]);
                        
        }

        ////////////////////////
        //mem0 to (residual and conv1)
        ////////////////////////
        image_mem0 = shared_buffer<bfloat16>::create({in_W, in_H, in_C}, 1, 6);
        location<buffer>(image_mem0) = location<kernel>(k_layer4_0_down[0])    + relative_offset({ .col_offset = -1 });  // 只支持列偏移
        num_buffers(image_mem0) = 2;
        for (int i = 0; i < 1; ++i) {
            write_access(image_mem0.in[i]) =
                tiling({.buffer_dimension = {in_W, in_H, in_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {in_W, in_H, in_C},              // 一次写 **1 行**
                    .offset           = {0, 0,  0},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{.dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(input_image, image_mem0.in[i]);}
        for (int i = 0; i < 3; ++i) {
            read_access(image_mem0.out[i]) 
            = tiling({.buffer_dimension   = {in_W, in_H, in_C},
                .tiling_dimension   = {in_W, in_H, in_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 }, { .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        for (int i = 0; i < 3; ++i) {
            read_access(image_mem0.out[i+3]) 
            = tiling({.buffer_dimension   = {in_W, in_H, in_C},
                .tiling_dimension   = {in_W, in_H, in_C/3},
                .offset             = {0, 0,  (in_C/3)*i},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 }, { .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
    
        connect<>(image_mem0.out[0], k_layer4_0_conv1[0].in[2]);
        connect<>(image_mem0.out[1], k_layer4_0_conv1[1].in[2]);
        connect<>(image_mem0.out[2], k_layer4_0_conv1[2].in[2]);
        connect<>(image_mem0.out[3], k_layer4_0_down[0].in[0]);
        connect<>(image_mem0.out[4], k_layer4_0_down[1].in[0]);
        connect<>(image_mem0.out[5], k_layer4_0_down[2].in[0]);

        ////////////////////////
        //0_conv1 to mem1
        ////////////////////////
        image_mem1 = shared_buffer<bfloat16>::create({out_W, out_H, out_C}, 3, 3);
        location<buffer>(image_mem1) = location<kernel>(k_layer4_0_down[0])    + relative_offset({ .col_offset = 0 });  // 只支持列偏移
        num_buffers(image_mem1) = 2;
        for (int i = 0; i < 3; ++i) {
            write_access(image_mem1.in[i]) =
                tiling({.buffer_dimension = {out_W, out_H, out_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {out_W, out_H, out_C/3},              // 一次写 **1 行**
                    .offset           = {0, 0,  (out_C/3)*i},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(k_layer4_0_conv1[i].out[0], image_mem1.in[i]);}
        for (int i = 0; i < 3; ++i) {
            read_access(image_mem1.out[i]) 
            = tiling({.buffer_dimension   = {out_W, out_H, out_C},
                .tiling_dimension   = {out_W, out_H, out_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        ////////////////////////
        //mem1 to 0_conv2
        ////////////////////////            
        connect<>(image_mem1.out[0], k_layer4_0_conv2[0].in[2]);
        connect<>(image_mem1.out[1], k_layer4_0_conv2[1].in[2]);
        connect<>(image_mem1.out[2], k_layer4_0_conv2[2].in[2]);

        ////////////////////////
        //0_residual to 0_conv2
        ////////////////////////
        for (int i = 0; i < 3; ++i) {
            connect<>(k_layer4_0_down[i].out[0], k_layer4_0_conv2[i].in[3]);
        }

        ////////////////////////
        //0_conv2 to mem2
        ////////////////////////
        image_mem2 = shared_buffer<bfloat16>::create({out_W, out_H, out_C}, 3, 1);
        location<buffer>(image_mem2) = location<kernel>(k_layer4_0_down[0])    + relative_offset({ .col_offset = 2 });  // 只支持列偏移
        num_buffers(image_mem2) = 2;
        for (int i = 0; i < 3; ++i) {
            write_access(image_mem2.in[i]) =
                tiling({.buffer_dimension = {out_W, out_H, out_C},   // 整块缓冲  W×H×C
                    .tiling_dimension = {out_W, out_H, out_C/3},              // 一次写 **1 行**
                    .offset           = {0, 0,  (out_C/3)*i},              // 第 i 个通道起点
                    .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});
            connect<>(k_layer4_0_conv2[i].out[0], image_mem2.in[i]);}
        for (int i = 0; i < 1; ++i) {
            read_access(image_mem2.out[i]) 
            = tiling({.buffer_dimension   = {out_W, out_H, out_C},
                .tiling_dimension   = {out_W, out_H, out_C},
                .offset             = {0, 0,  0},
                .tile_traversal     = {{ .dimension = 0, .stride = 0, .wrap = 1 },{ .dimension = 1, .stride = 0, .wrap = 1 },{ .dimension = 2, .stride =  0, .wrap = 1 }},});}
        
        ////////////////////////
        //mem2 to output
        ////////////////////////
        connect<>(image_mem2.out[0], output[0]);
    }
};




template<
    int  X0  = 16, int  Y0  = 3,            // mem_to_4 起点
    int  X1  = 0, int  Y1  = 0,            // stage-2  起点
    int  NK  = 1                            // stage-2 个数
>
class layer5_head_1out : public adf::graph
{
private:
    kernel                   k_head;   
public:
 
    adf::port<input>  input_image;
    adf::port<input>  input_weight;
    

    adf::port<output> output[1];
    layer5_head_1out()
    {   
        //0_conv1
        k_head = kernel::create_object<layer5_head>(0);          
        source(k_head) = "layer5_head.cpp";
        runtime<ratio>(k_head) = 0.9;
        location<kernel>(k_head) = tile(X0 , Y0);
        location<stack >(k_head) = location<kernel>(k_head);
        location<buffer>(k_head.in[0]) = location<kernel>(k_head);
        
        
        connect<>(input_image, k_head.in[1]);
        connect<>(input_weight, k_head.in[0]);
        
        connect<>(k_head.out[0], output[0]);
    }
};




template<
    int  X0  = 16, int  Y0  = 3,            // mem_to_4 起点
    int  X1  = 0, int  Y1  = 0,            // stage-2  起点
    int  NK  = 1                            // stage-2 个数
>
class singletile_test : public adf::graph
{
private:
    kernel                   k_test;   
public:
 
    adf::port<input>  input_image;
    //adf::port<input>  input_weight;
    //adf::port<input>  input_residual;

    adf::port<output> output[1];
    singletile_test()
    {   
        //0_conv1
        k_test = kernel::create_object<fft64_test>(0);          
        source(k_test) = "fft64_test.cpp";
        runtime<ratio>(k_test) = 0.9;
        location<kernel>(k_test) = tile(X0 , Y0);
        location<stack >(k_test) = location<kernel>(k_test);
        //location<buffer>(k_test.in[0]) = location<kernel>(k_test);  
        
        connect<>(input_image, k_test.in[0]);
        //connect<>(input_weight, k_test.in[0]);
        //connect<>(input_residual, k_test.in[2]);
       
        connect<>(k_test.out[0], output[0]);
    }
};






#endif //__SSCA_SYS_H__

