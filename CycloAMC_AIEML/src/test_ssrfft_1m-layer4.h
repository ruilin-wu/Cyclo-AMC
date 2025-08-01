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


    layer4_block0_1memout<28-1,0,0,0,1> layer4_graph_0;
    layer4_block1_1memout<28+3+1-1,0,0,0,1> layer4_graph_1;
    
    input_plio      layer4_0_conv1_weight_0[3];
    input_plio      layer4_0_conv2_weight_0[3];
    input_plio      layer4_0_conv1_weight_1[3];
    input_plio      layer4_0_conv2_weight_1[3];
    input_plio      layer4_0_down_weight[3];

    
    input_plio      layer4_1_conv1_weight_0[3];
    input_plio      layer4_1_conv2_weight_0[3];
    input_plio      layer4_1_conv1_weight_1[3];
    input_plio      layer4_1_conv2_weight_1[3];

    
    input_plio      layer4_input0;
    output_plio     layer4_out;
    dut(){
        //layer4-conv1        
        for ( int i = 0; i < 3; i++)
        {
        layer4_0_conv1_weight_0[i] = input_plio::create("layer4_0_conv1_weight_0"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight1.txt", PLFreq);
        layer4_0_conv1_weight_1[i] = input_plio::create("layer4_0_conv1_weight_1"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight1.txt", PLFreq);
        layer4_0_down_weight[i] = input_plio::create("layer4_0_down_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        layer4_0_conv2_weight_0[i] = input_plio::create("layer4_0_conv2_weight_0"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        layer4_0_conv2_weight_1[i] = input_plio::create("layer4_0_conv2_weight_1"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        connect<>(layer4_0_conv1_weight_0[i].out[0], layer4_graph_0.input_weight_0_conv1[i]);
        connect<>(layer4_0_conv2_weight_0[i].out[0], layer4_graph_0.input_weight_0_conv2[i]);
        connect<>(layer4_0_conv1_weight_1[i].out[0], layer4_graph_0.input_weight_1_conv1[i]);
        connect<>(layer4_0_conv2_weight_1[i].out[0], layer4_graph_0.input_weight_1_conv2[i]);
        connect<>(layer4_0_down_weight[i].out[0], layer4_graph_0.input_weight_0_down[i]);
        } 
        for ( int i = 0; i < 3; i++)
        {
        layer4_1_conv1_weight_0[i] = input_plio::create("layer4_1_conv1_weight_0"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight1.txt", PLFreq);
        layer4_1_conv1_weight_1[i] = input_plio::create("layer4_1_conv1_weight_1"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight1.txt", PLFreq);
        
        layer4_1_conv2_weight_0[i] = input_plio::create("layer4_1_conv2_weight_0"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        layer4_1_conv2_weight_1[i] = input_plio::create("layer4_1_conv2_weight_1"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        connect<>(layer4_1_conv1_weight_0[i].out[0], layer4_graph_1.input_weight_0_conv1[i]);
        connect<>(layer4_1_conv1_weight_1[i].out[0], layer4_graph_1.input_weight_1_conv1[i]);
        
        connect<>(layer4_1_conv2_weight_0[i].out[0], layer4_graph_1.input_weight_0_conv2[i]);
        connect<>(layer4_1_conv2_weight_1[i].out[0], layer4_graph_1.input_weight_1_conv2[i]);
        } 
        {
        layer4_input0 = input_plio::create("layer4_input0", plio_32_bits, "data_input/layer4datain.txt", PLFreq);
        layer4_out = output_plio::create("layer4_out", plio_32_bits, "data_output/layer4_out.txt", PLFreq);         
        connect<>(layer4_input0.out[0], layer4_graph_0.input_image);      
        connect<>(layer4_graph_0.output[0],layer4_graph_1.input_image);       
        connect<>(layer4_graph_1.output[0],layer4_out.in[0]); 
        }
	};
}; // end of class

