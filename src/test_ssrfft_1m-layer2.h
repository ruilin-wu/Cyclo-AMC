//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
/*
Check AIE-PRE-MAPPER has run: 1 error
ERROR: [AIE-PRE-MAPPER-6] Insufficient shim routing capacity. Design has outgoing PLIO channel usage of 128 and maximum that can be routed of 112.
AIE Initial Checker Successful
ERROR: [aiecompiler 77-6231] ###UDM Placer Did NOT Finish Successfully
INFO: [aiecompiler 77-6439] Run completed. Find additional information in:
        Guidance: /home/ruilin/aie/aie-fft/build.hw/Work/reports/guidance.html

INFO: [aiecompiler 77-6440] Use the vitis_analyzer tool to visualize and navigate the relevant reports. Run: 
        vitis_analyzer /home/ruilin/aie/aie-fft/build.hw/Work/test_ssrfft_1m.aiecompile_summary

Compilation Failed
(WARNING:138, CRITICAL-WARNING:0, ERROR:1)
*/
#include "fam_sys.h"

#define PLFreq 1000 //1000MHZ

using namespace adf ;
class dut: public graph {
public:
    //sim time:543.13 s
    //mem_to_4_graph_test<13,2,0> mem_to_4;
    static constexpr int NumOutput = 1;
    static constexpr int Numinput = 1;



    layer2_block0_1memout<17,0,0,0,NumOutput> layer2_graph_0;
    layer2_block1_1memout<20,0,0,0,NumOutput> layer2_graph_1;
    //layer1_block0_1memout<20,0,0,0,NumOutput> layer1_graph_1;

    input_plio      layer2_0_conv1_weight[3];
    input_plio      layer2_0_conv2_weight[3];
    input_plio      layer2_0_down_weight[3];
    input_plio      layer2_1_conv1_weight[3];
    input_plio      layer2_1_conv2_weight[3];
    input_plio      layer2_1_down_weight[3];
    input_plio      layer2_input0;
    output_plio     layer2_out;
    dut(){

        //layer1-conv1
        
        for ( int i = 0; i < 3; i++)
        {
        layer2_0_conv1_weight[i] = input_plio::create("layer2_0_conv1_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight1.txt", PLFreq);
        layer2_0_down_weight[i] = input_plio::create("layer2_0_down_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        layer2_0_conv2_weight[i] = input_plio::create("layer2_0_conv2_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        connect<>(layer2_0_conv1_weight[i].out[0], layer2_graph_0.input_weight_0_conv1[i]);
        connect<>(layer2_0_down_weight[i].out[0], layer2_graph_0.input_weight_0_down[i]);
        connect<>(layer2_0_conv2_weight[i].out[0], layer2_graph_0.input_weight_0_conv2[i]);
        }                        
        for ( int i = 0; i < 3; i++)
        {
        layer2_1_conv1_weight[i] = input_plio::create("layer2_1_conv1_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight1.txt", PLFreq);
        layer2_1_down_weight[i] = input_plio::create("layer2_1_down_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        layer2_1_conv2_weight[i] = input_plio::create("layer2_1_conv2_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        connect<>(layer2_1_conv1_weight[i].out[0], layer2_graph_1.input_weight_1_conv1[i]);
        connect<>(layer2_1_down_weight[i].out[0], layer2_graph_1.input_weight_1_down[i]);
        connect<>(layer2_1_conv2_weight[i].out[0], layer2_graph_1.input_weight_1_conv2[i]);
        } 
        {
        layer2_input0 = input_plio::create("layer2_input0", plio_32_bits, "data_input/layer2datain.txt", PLFreq);
        layer2_out = output_plio::create("layer2_out", plio_32_bits, "data_output/layer2_out.txt", PLFreq);         
        connect<>(layer2_input0.out[0], layer2_graph_0.input_image);      
        connect<>(layer2_graph_0.output[0],layer2_graph_1.input_image);       
        connect<>(layer2_graph_1.output[0],layer2_out.in[0]); 
        }
	};
}; // end of class

