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

    fam_top_test<7,4,0,0,NumOutput> fam;
    input_plio      fam_input;
    

    stem<16,0,0,0,NumOutput> stem_graph;
    input_plio      stem_weight;
    
    output_plio     stem_out;

    dut(){
        //fam
        fam_input   = input_plio::create("fam_input", plio_32_bits, "data_input/FAMDataIn.txt", PLFreq);
        connect<>(fam_input.out[0], fam.input);
        
        //layer0-stem
        connect<>(fam.output[0],stem_graph.input_image[0]);
        stem_weight = input_plio::create("stem_weight", plio_32_bits, "data_input/txt_weights/stem.0.txt", PLFreq);        
        stem_out = output_plio::create("stem_out", plio_32_bits, "data_output/STEMOut.txt", PLFreq);
        connect<>(stem_weight.out[0], stem_graph.input_weight[0]);
        connect<>(stem_graph.output[0],stem_out.in[0]); 

        //layer1-conv1



	};
}; // end of class

