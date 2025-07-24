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
        //layer0-stem
        stem<7+3,0,0,0,1> stem_graph;
        input_plio      stem_weight;
        input_plio      stem_image; //fam_out
        output_plio     stem_out;

        //layer1
        layer1_block0_1memout<7+5,0,0,0,1> layer1_graph_0;
        layer1_block0_1memout<7+3+5,0,0,0,1> layer1_graph_1;    
        input_plio      layer1_0_conv1_weight[3];
        input_plio      layer1_0_conv2_weight[3];
        input_plio      layer1_0_down_weight[3];
        input_plio      layer1_1_conv1_weight[3];
        input_plio      layer1_1_conv2_weight[3];
        input_plio      layer1_1_down_weight[3];
        input_plio      layer1_input0;
        output_plio     layer1_out;

        //layer2
        layer2_block0_1memout<16+2,0,0,0,1> layer2_graph_0;
        layer2_block1_1memout<19+2,0,0,0,1> layer2_graph_1;
        input_plio      layer2_0_conv1_weight[3];
        input_plio      layer2_0_conv2_weight[3];
        input_plio      layer2_0_down_weight[3];
        input_plio      layer2_1_conv1_weight[3];
        input_plio      layer2_1_conv2_weight[3];
        input_plio      layer2_1_down_weight[3];
        input_plio      layer2_input0;
        output_plio     layer2_out;


        //layer3
        layer3_block0_1memout<24,0,0,0,1> layer3_graph_0;
        layer3_block1_1memout<27,0,0,0,1> layer3_graph_1;
        input_plio      layer3_0_conv1_weight[3];
        input_plio      layer3_0_conv2_weight[3];
        input_plio      layer3_0_down_weight[3];
        input_plio      layer3_1_conv1_weight[3];
        input_plio      layer3_1_conv2_weight[3];
        input_plio      layer3_1_down_weight[3];
        input_plio      layer3_input0;
        output_plio     layer3_out;

        //layer4
        layer4_block0_1memout<31,0,0,0,1> layer4_graph_0;
        layer4_block1_1memout<34,0,0,0,1> layer4_graph_1;
        input_plio      layer4_0_conv1_weight[3];
        input_plio      layer4_0_conv2_weight[3];
        input_plio      layer4_0_down_weight[3];
        input_plio      layer4_1_conv1_weight[3];
        input_plio      layer4_1_conv2_weight[3];
        input_plio      layer4_1_down_weight[3];
        input_plio      layer4_input0;
        output_plio     layer4_out;

        //layer5-head
        layer5_head_1out<23,3,0,0,0> head_graph;
        input_plio      layer5_image;
        input_plio      layer5_weight;
        output_plio     layer5_out;

    dut(){
        ////////////////////////layer0//////////////////////
        {
        stem_image = input_plio::create("stem_image", plio_32_bits, "data_input/StemDataIn.txt", PLFreq);
        stem_weight = input_plio::create("stem_weight", plio_32_bits, "data_input/txt_weights/stem.0.txt", PLFreq);
        stem_out = output_plio::create("stem_out", plio_32_bits, "data_output/STEMOut.txt", PLFreq);
        connect<>(stem_image.out[0],stem_graph.input_image[0]);
        connect<>(stem_weight.out[0], stem_graph.input_weight[0]);
        connect<>(stem_graph.output[0],stem_out.in[0]);
        }

        ////////////////////////layer1//////////////////////
        for ( int i = 0; i < 3; i++)
        {
        layer1_0_conv1_weight[i] = input_plio::create("layer1_0_conv1_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight1.txt", PLFreq);
        layer1_0_down_weight[i] = input_plio::create("layer1_0_down_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        layer1_0_conv2_weight[i] = input_plio::create("layer1_0_conv2_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        connect<>(layer1_0_conv1_weight[i].out[0], layer1_graph_0.input_weight_0_conv1[i]);
        connect<>(layer1_0_down_weight[i].out[0], layer1_graph_0.input_weight_0_down[i]);
        connect<>(layer1_0_conv2_weight[i].out[0], layer1_graph_0.input_weight_0_conv2[i]);
        }
        for ( int i = 0; i < 3; i++)
        {
        layer1_1_conv1_weight[i] = input_plio::create("layer1_1_conv1_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight1.txt", PLFreq);
        layer1_1_down_weight[i] = input_plio::create("layer1_1_down_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        layer1_1_conv2_weight[i] = input_plio::create("layer1_1_conv2_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        connect<>(layer1_1_conv1_weight[i].out[0], layer1_graph_1.input_weight_0_conv1[i]);
        connect<>(layer1_1_down_weight[i].out[0], layer1_graph_1.input_weight_0_down[i]);
        connect<>(layer1_1_conv2_weight[i].out[0], layer1_graph_1.input_weight_0_conv2[i]);
        }        
        layer1_input0 = input_plio::create("layer1_input0", plio_32_bits, "data_input/layer1datain.txt", PLFreq);
        layer1_out = output_plio::create("layer1_out", plio_32_bits, "data_output/layer1_out.txt", PLFreq);        
        connect<>(layer1_input0.out[0], layer1_graph_0.input_image);
        connect<>(layer1_graph_0.output[0],layer1_graph_1.input_image);       
        connect<>(layer1_graph_1.output[0],layer1_out.in[0]); 
        
        ////////////////////////layer2//////////////////////
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


        ////////////////////////layer3//////////////////////
        for ( int i = 0; i < 3; i++)
        {
        layer3_0_conv1_weight[i] = input_plio::create("layer3_0_conv1_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight1.txt", PLFreq);
        layer3_0_down_weight[i] = input_plio::create("layer3_0_down_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        layer3_0_conv2_weight[i] = input_plio::create("layer3_0_conv2_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        connect<>(layer3_0_conv1_weight[i].out[0], layer3_graph_0.input_weight_0_conv1[i]);
        connect<>(layer3_0_down_weight[i].out[0], layer3_graph_0.input_weight_0_down[i]);
        connect<>(layer3_0_conv2_weight[i].out[0], layer3_graph_0.input_weight_0_conv2[i]);
        } 
        for ( int i = 0; i < 3; i++)
        {
        layer3_1_conv1_weight[i] = input_plio::create("layer3_1_conv1_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight1.txt", PLFreq);
        layer3_1_down_weight[i] = input_plio::create("layer3_1_down_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        layer3_1_conv2_weight[i] = input_plio::create("layer3_1_conv2_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        connect<>(layer3_1_conv1_weight[i].out[0], layer3_graph_1.input_weight_0_conv1[i]);
        connect<>(layer3_1_down_weight[i].out[0], layer3_graph_1.input_weight_0_down[i]);
        connect<>(layer3_1_conv2_weight[i].out[0], layer3_graph_1.input_weight_0_conv2[i]);
        } 
        {
        layer3_input0 = input_plio::create("layer3_input0", plio_32_bits, "data_input/layer3datain.txt", PLFreq);
        layer3_out = output_plio::create("layer3_out", plio_32_bits, "data_output/layer3_out.txt", PLFreq);         
        connect<>(layer3_input0.out[0], layer3_graph_0.input_image);      
        connect<>(layer3_graph_0.output[0],layer3_graph_1.input_image);       
        connect<>(layer3_graph_1.output[0],layer3_out.in[0]); 
        }     

        ////////////////////////layer4//////////////////////
        for ( int i = 0; i < 3; i++)
        {
        layer4_0_conv1_weight[i] = input_plio::create("layer4_0_conv1_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight1.txt", PLFreq);
        layer4_0_down_weight[i] = input_plio::create("layer4_0_down_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        layer4_0_conv2_weight[i] = input_plio::create("layer4_0_conv2_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        connect<>(layer4_0_conv1_weight[i].out[0], layer4_graph_0.input_weight_0_conv1[i]);
        connect<>(layer4_0_down_weight[i].out[0], layer4_graph_0.input_weight_0_down[i]);
        connect<>(layer4_0_conv2_weight[i].out[0], layer4_graph_0.input_weight_0_conv2[i]);
        } 
        for ( int i = 0; i < 3; i++)
        {
        layer4_1_conv1_weight[i] = input_plio::create("layer4_1_conv1_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight1.txt", PLFreq);
        layer4_1_down_weight[i] = input_plio::create("layer4_1_down_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        layer4_1_conv2_weight[i] = input_plio::create("layer4_1_conv2_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/weight2.txt", PLFreq);
        connect<>(layer4_1_conv1_weight[i].out[0], layer4_graph_1.input_weight_0_conv1[i]);
        connect<>(layer4_1_down_weight[i].out[0], layer4_graph_1.input_weight_0_down[i]);
        connect<>(layer4_1_conv2_weight[i].out[0], layer4_graph_1.input_weight_0_conv2[i]);
        } 
        {
        layer4_input0 = input_plio::create("layer4_input0", plio_32_bits, "data_input/layer4datain.txt", PLFreq);
        layer4_out = output_plio::create("layer4_out", plio_32_bits, "data_output/layer4_out.txt", PLFreq);         
        connect<>(layer4_input0.out[0], layer4_graph_0.input_image);      
        connect<>(layer4_graph_0.output[0],layer4_graph_1.input_image);       
        connect<>(layer4_graph_1.output[0],layer4_out.in[0]); 
        }


        


        ////////////////////////layer5//////////////////////
        layer5_image = input_plio::create("layer5_image", plio_32_bits, "data_input/layer5datain.txt", PLFreq);
        layer5_weight = input_plio::create("layer5_weight", plio_32_bits, "data_input/txt_weights/weight1.txt", PLFreq);
        layer5_out = output_plio::create("layer5_out", plio_32_bits, "data_output/layer5_out.txt", PLFreq);        
        connect<>(layer5_image.out[0], head_graph.input_image);
        connect<>(layer5_weight.out[0], head_graph.input_weight);
        connect<>(head_graph.output[0], layer5_out.in[0]);


        
	};
}; // end of class


