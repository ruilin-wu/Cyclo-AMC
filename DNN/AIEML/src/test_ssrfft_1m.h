//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "fam_sys.h"

#define PLFreq 625 //1000MHZ

using namespace adf ;
class dut: public graph {
public: 
        //fam        
        fam_top<3,4,0> fam_graph;
        input_plio      input_image;
        output_plio     fam_output[1];
        

        //layer0-stem
        stem<6,0,0,0,1> stem_graph;
        input_plio      stem_weight;
        input_plio      stem_image; 


        //layer1
        layer1_block0_1memout<8,0,0,0,1> layer1_graph_0;
        layer1_block0_1memout<11,0,0,0,1> layer1_graph_1;    
        input_plio      layer1_0_conv1_weight[3];
        input_plio      layer1_0_conv2_weight[3];
        input_plio      layer1_0_down_weight[3];
        input_plio      layer1_1_conv1_weight[3];
        input_plio      layer1_1_conv2_weight[3];
        input_plio      layer1_1_down_weight[3];


        //layer2
        layer2_block0_1memout<14,0,0,0,1> layer2_graph_0;
        layer2_block1_1memout<17,0,0,0,1> layer2_graph_1;
        input_plio      layer2_0_conv1_weight[3];
        input_plio      layer2_0_conv2_weight[3];
        input_plio      layer2_0_down_weight[3];
        input_plio      layer2_1_conv1_weight[3];
        input_plio      layer2_1_conv2_weight[3];
        input_plio      layer2_1_down_weight[3];
        


        //layer3
        layer3_block0_1memout<20,0,0,0,1> layer3_graph_0;
        layer3_block1_1memout<23,0,0,0,1> layer3_graph_1;
        input_plio      layer3_0_conv1_weight[3];
        input_plio      layer3_0_conv2_weight[3];
        input_plio      layer3_0_down_weight[3];
        input_plio      layer3_1_conv1_weight[3];
        input_plio      layer3_1_conv2_weight[3];
        input_plio      layer3_1_down_weight[3];
        

        //layer4
        layer4_block0_1memout<27,0,0,0,1> layer4_graph_0;
        layer4_block1_1memout<31,0,0,0,1> layer4_graph_1;
        input_plio      layer4_0_conv1_weight_0[3];
        input_plio      layer4_0_conv2_weight_0[3];
        input_plio      layer4_0_conv1_weight_1[3];
        input_plio      layer4_0_conv2_weight_1[3];
        input_plio      layer4_0_down_weight[3];

        
        input_plio      layer4_1_conv1_weight_0[3];
        input_plio      layer4_1_conv2_weight_0[3];
        input_plio      layer4_1_conv1_weight_1[3];
        input_plio      layer4_1_conv2_weight_1[3];
        

        //layer5-head
        layer5_head_1out<33,0,0,0,0> head_graph;
        input_plio      layer5_weight;
        output_plio     layer5_out;

        //test
        output_plio     layer0_test_out;
        output_plio     layer1_test_out;
        output_plio     layer2_test_out;
        output_plio     layer3_test_out;
        output_plio     layer4_test_out;
        output_plio     layer5_test_out;

    dut(){
        ////////////////////////fam//////////////////////
        
        input_image = input_plio::create("input_image", plio_32_bits, "data_input/fam_in.txt", PLFreq);        
        fam_output[0] = output_plio::create("fam_output", plio_32_bits, "data_output/fam_out.txt", PLFreq);        
        connect<>(input_image.out[0], fam_graph.input);     
        connect<>(fam_graph.output[0], fam_output[0].in[0]);
        

        ////////////////////////layer0//////////////////////
        {
        stem_image = input_plio::create("stem_image", plio_32_bits, "data_input/StemDataIn.txt", PLFreq);
        stem_weight = input_plio::create("stem_weight", plio_32_bits, "data_input/txt_weights/stem_0.txt", PLFreq);
        connect<>(stem_image.out[0],stem_graph.input_image[0]);
        connect<>(stem_weight.out[0], stem_graph.input_weight[0]);
        
        }

        ////////////////////////layer1//////////////////////
        for ( int i = 0; i < 3; i++)
        {
        layer1_0_conv1_weight[i] = input_plio::create("layer1_0_conv1_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer1_0_conv1_weight_"+ std::to_string(i)+ ".txt", PLFreq);
        layer1_0_conv2_weight[i] = input_plio::create("layer1_0_conv2_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer1_0_conv2_weight_"+ std::to_string(i)+ ".txt", PLFreq);
        connect<>(layer1_0_conv1_weight[i].out[0], layer1_graph_0.input_weight_0_conv1[i]);
        connect<>(layer1_0_conv2_weight[i].out[0], layer1_graph_0.input_weight_0_conv2[i]);
        }
        for ( int i = 0; i < 3; i++)
        {
        layer1_1_conv1_weight[i] = input_plio::create("layer1_1_conv1_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer1_1_conv1_weight_"+ std::to_string(i)+ ".txt", PLFreq);
        layer1_1_conv2_weight[i] = input_plio::create("layer1_1_conv2_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer1_1_conv2_weight_"+ std::to_string(i)+ ".txt", PLFreq);
        connect<>(layer1_1_conv1_weight[i].out[0], layer1_graph_1.input_weight_0_conv1[i]);
        connect<>(layer1_1_conv2_weight[i].out[0], layer1_graph_1.input_weight_0_conv2[i]);
        }        
        connect<>(stem_graph.output[0], layer1_graph_0.input_image);
        connect<>(layer1_graph_0.output[0],layer1_graph_1.input_image);       
        

        
        ////////////////////////layer2//////////////////////
        for ( int i = 0; i < 3; i++)
        {
        layer2_0_conv1_weight[i] = input_plio::create("layer2_0_conv1_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer2_0_conv1_weight_"+ std::to_string(i)+ ".txt", PLFreq);
        layer2_0_down_weight[i] = input_plio::create("layer2_0_down_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer2_0_down_0_weight_"+ std::to_string(i)+ ".txt", PLFreq);
        layer2_0_conv2_weight[i] = input_plio::create("layer2_0_conv2_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer2_0_conv2_weight_"+ std::to_string(i)+ ".txt", PLFreq);
        connect<>(layer2_0_conv1_weight[i].out[0], layer2_graph_0.input_weight_0_conv1[i]);
        connect<>(layer2_0_down_weight[i].out[0], layer2_graph_0.input_weight_0_down[i]);
        connect<>(layer2_0_conv2_weight[i].out[0], layer2_graph_0.input_weight_0_conv2[i]);
        }                        
        for ( int i = 0; i < 3; i++)
        {
        layer2_1_conv1_weight[i] = input_plio::create("layer2_1_conv1_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer2_1_conv1_weight_"+ std::to_string(i)+ ".txt", PLFreq);
        layer2_1_conv2_weight[i] = input_plio::create("layer2_1_conv2_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer2_1_conv2_weight_"+ std::to_string(i)+ ".txt", PLFreq);
        connect<>(layer2_1_conv1_weight[i].out[0], layer2_graph_1.input_weight_1_conv1[i]);
        connect<>(layer2_1_conv2_weight[i].out[0], layer2_graph_1.input_weight_1_conv2[i]);
        } 
        {
        connect<>(layer1_graph_1.output[0],layer2_graph_0.input_image);      
        connect<>(layer2_graph_0.output[0],layer2_graph_1.input_image);                      
        }


        ////////////////////////layer3//////////////////////
        for ( int i = 0; i < 3; i++)
        {
        layer3_0_conv1_weight[i] = input_plio::create("layer3_0_conv1_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer3_0_conv1_weight_"+ std::to_string(i)+ ".txt", PLFreq);
        layer3_0_down_weight[i] = input_plio::create("layer3_0_down_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer3_0_down_0_weight_"+ std::to_string(i)+ ".txt", PLFreq);
        layer3_0_conv2_weight[i] = input_plio::create("layer3_0_conv2_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer3_0_conv2_weight_"+ std::to_string(i)+ ".txt", PLFreq);
        connect<>(layer3_0_conv1_weight[i].out[0], layer3_graph_0.input_weight_0_conv1[i]);
        connect<>(layer3_0_down_weight[i].out[0], layer3_graph_0.input_weight_0_down[i]);
        connect<>(layer3_0_conv2_weight[i].out[0], layer3_graph_0.input_weight_0_conv2[i]);
        } 
        for ( int i = 0; i < 3; i++)
        {
        layer3_1_conv1_weight[i] = input_plio::create("layer3_1_conv1_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer3_1_conv1_weight_"+ std::to_string(i)+ ".txt", PLFreq);
        layer3_1_conv2_weight[i] = input_plio::create("layer3_1_conv2_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer3_1_conv2_weight_"+ std::to_string(i)+ ".txt", PLFreq);
        connect<>(layer3_1_conv1_weight[i].out[0], layer3_graph_1.input_weight_0_conv1[i]);
        connect<>(layer3_1_conv2_weight[i].out[0], layer3_graph_1.input_weight_0_conv2[i]);
        } 
        {
        connect<>(layer2_graph_1.output[0], layer3_graph_0.input_image);      
        connect<>(layer3_graph_0.output[0],layer3_graph_1.input_image);    
              
        }     

        ////////////////////////layer4//////////////////////
        for ( int i = 0; i < 3; i++)
        {
        layer4_0_conv1_weight_0[i] = input_plio::create("layer4_0_conv1_weight_0"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer4_0_conv1_weight_0_"+ std::to_string(i)+ ".txt", PLFreq);
        layer4_0_conv1_weight_1[i] = input_plio::create("layer4_0_conv1_weight_1"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer4_0_conv1_weight_1_"+ std::to_string(i)+ ".txt", PLFreq);
        layer4_0_down_weight[i] = input_plio::create("layer4_0_down_weight"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer4_0_down_0_weight_"+ std::to_string(i)+ ".txt", PLFreq);
        layer4_0_conv2_weight_0[i] = input_plio::create("layer4_0_conv2_weight_0"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer4_0_conv2_weight_0_"+ std::to_string(i)+ ".txt", PLFreq);
        layer4_0_conv2_weight_1[i] = input_plio::create("layer4_0_conv2_weight_1"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer4_0_conv2_weight_1_"+ std::to_string(i)+ ".txt", PLFreq);
        connect<>(layer4_0_conv1_weight_0[i].out[0], layer4_graph_0.input_weight_0_conv1[i]);
        connect<>(layer4_0_conv2_weight_0[i].out[0], layer4_graph_0.input_weight_0_conv2[i]);
        connect<>(layer4_0_conv1_weight_1[i].out[0], layer4_graph_0.input_weight_1_conv1[i]);
        connect<>(layer4_0_conv2_weight_1[i].out[0], layer4_graph_0.input_weight_1_conv2[i]);
        connect<>(layer4_0_down_weight[i].out[0], layer4_graph_0.input_weight_0_down[i]);
        } 
        for ( int i = 0; i < 3; i++)
        {
        layer4_1_conv1_weight_0[i] = input_plio::create("layer4_1_conv1_weight_0"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer4_1_conv1_weight_0_"+ std::to_string(i)+ ".txt", PLFreq);
        layer4_1_conv1_weight_1[i] = input_plio::create("layer4_1_conv1_weight_1"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer4_1_conv1_weight_1_"+ std::to_string(i)+ ".txt", PLFreq);
        
        layer4_1_conv2_weight_0[i] = input_plio::create("layer4_1_conv2_weight_0"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer4_1_conv2_weight_0_"+ std::to_string(i)+ ".txt", PLFreq);
        layer4_1_conv2_weight_1[i] = input_plio::create("layer4_1_conv2_weight_1"+ std::to_string(i), plio_32_bits, "data_input/txt_weights/layer4_1_conv2_weight_1_"+ std::to_string(i)+ ".txt", PLFreq);
        connect<>(layer4_1_conv1_weight_0[i].out[0], layer4_graph_1.input_weight_0_conv1[i]);
        connect<>(layer4_1_conv1_weight_1[i].out[0], layer4_graph_1.input_weight_1_conv1[i]);
        
        connect<>(layer4_1_conv2_weight_0[i].out[0], layer4_graph_1.input_weight_0_conv2[i]);
        connect<>(layer4_1_conv2_weight_1[i].out[0], layer4_graph_1.input_weight_1_conv2[i]);
        } 
        {
        connect<>(layer3_graph_1.output[0], layer4_graph_0.input_image);    
        connect<>(layer4_graph_0.output[0],layer4_graph_1.input_image);               
        
              
        }

        ////////////////////////layer5//////////////////////
        layer5_weight = input_plio::create("layer5_weight", plio_32_bits, "data_input/txt_weights/head_3_weight.txt", PLFreq);
        layer5_out = output_plio::create("layer5_out", plio_32_bits, "data_output/layer5_out.txt", PLFreq); 
        connect<>(layer4_graph_1.output[0],head_graph.input_image);      
        connect<>(layer5_weight.out[0], head_graph.input_weight);
        connect<>(head_graph.output[0], layer5_out.in[0]);



        ////////////////////////test//////////////////////
        layer0_test_out = output_plio::create("layer0_test_out", plio_32_bits, "data_output/layer0_test_out.txt", PLFreq);
        layer1_test_out = output_plio::create("layer1_test_out", plio_32_bits, "data_output/layer1_test_out.txt", PLFreq);
        layer2_test_out = output_plio::create("layer2_test_out", plio_32_bits, "data_output/layer2_test_out.txt", PLFreq);
        layer3_test_out = output_plio::create("layer3_test_out", plio_32_bits, "data_output/layer3_test_out.txt", PLFreq);
        layer4_test_out = output_plio::create("layer4_test_out", plio_32_bits, "data_output/layer4_test_out.txt", PLFreq);
        layer5_test_out = output_plio::create("layer5_test_out", plio_32_bits, "data_output/layer5_test_out.txt", PLFreq);
        connect<>(stem_graph.output[0], layer0_test_out.in[0]);
        connect<>(layer1_graph_1.output[0], layer1_test_out.in[0]);
        connect<>(layer2_graph_1.output[0], layer2_test_out.in[0]);
        connect<>(layer3_graph_1.output[0], layer3_test_out.in[0]);
        connect<>(layer4_graph_1.output[0], layer4_test_out.in[0]);
        connect<>(head_graph.output[0], layer5_test_out.in[0]);
        
	};
}; // end of class


