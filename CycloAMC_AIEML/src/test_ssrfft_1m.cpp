//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "test_ssrfft_1m.h"


dut dut;

int main(void) {
	dut.init();
    dut.run(1) ;
	dut.end() ;
    return 0 ;
}
