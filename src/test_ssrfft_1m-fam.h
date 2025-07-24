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

    fam_top<2,4,0> test_graph;

    input_plio      input_image;
    output_plio     output[1];


    dut(){
        input_image = input_plio::create("input_image", plio_32_bits, "data_input/fam_in.txt", PLFreq);        
        output[0] = output_plio::create("output1", plio_32_bits, "data_output/fam_out1.txt", PLFreq);        
        connect<>(input_image.out[0], test_graph.input);     
        connect<>(test_graph.output[0], output[0].in[0]);
        
	};
}; // end of class

