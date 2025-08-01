#
# AMD AIE FFT Example Makefile
#
# 该 Makefile 参考了用户给出的 ConvNet 版本, 并根据 aie-fft/src 目录中的 FFT 源文件进行简化调整。
#
# Author: AI Assistant
################################################################################
# 可配变量
################################################################################
# 仅进行 AIE 仿真(不生成硬件映像)
AIE_SIM_ONLY ?= false
# 打开 FIFO 深度检查
SIM_FIFO     ?= false

################################################################################
# 源文件与目录
################################################################################
SRC_DIR   := src
MY_APP    := test_ssrfft_1m

# 头文件搜索路径（含 src/funcs 与 src/fam_kernels）
USER_INCLUDES := --include=$(CURDIR)/$(SRC_DIR) \
                 --include=$(CURDIR)/$(SRC_DIR)/funcs \
                 --include=$(CURDIR)/$(SRC_DIR)/fam_kernels \
				 --include=$(CURDIR)/$(SRC_DIR)/nn_kernels

# 主源文件 (使用绝对路径)
MY_SOURCES := $(CURDIR)/$(SRC_DIR)/test_ssrfft_1m.cpp

################################################################################
# 生成目录与产物
################################################################################
BUILD_DIR := build.hw              # ← 目标输出目录（与 src 同级）
AIE_OUTPUT := $(BUILD_DIR)/libadf.a

################################################################################
# 平台信息
################################################################################
PLATFORM_USE := xilinx_vek280_base_202420_1
PLATFORM ?= $(PLATFORM_REPO_PATHS)/$(PLATFORM_USE)/$(PLATFORM_USE).xpfm

################################################################################
# 编译选项
################################################################################
CHECK_FIFO := --aie.evaluate-fifo-depth --aie.Xrouter=disablePathBalancing

AIE_FLAGS := $(USER_INCLUDES) \
             --platform=$(PLATFORM) \
			 --aie.Xelfgen="-j2"  \
             $(MY_SOURCES) \
             --aie.output-archive=$(AIE_OUTPUT) \
             --aie.swfifo-threshold 40

ifeq ($(SIM_FIFO), true)
	AIE_FLAGS += $(CHECK_FIFO)
endif
ifeq ($(AIE_SIM_ONLY), true)
	AIE_FLAGS += --aie.Xpreproc="-DAIE_SIM_ONLY"
endif

################################################################################
# 通用伪目标
################################################################################
.PHONY: help clean all compile sim sim1 x86sim profile trace analyze summary

help::
	@echo "Makefile Usage:"
	@echo "  make all       : 编译并生成 libadf.a"
	@echo "  make sim       : 运行 AIE 仿真器 (带 profile+VCD)"
	@echo "  make sim1      : 运行 AIE 仿真器 (纯功能仿真)"
	@echo "  make x86sim    : 运行功能级 x86 仿真"
	@echo "  make profile   : 同 sim"
	@echo "  make trace     : 同 sim (已含 VCD)"
	@echo "  make analyze   : 打开 Vitis Analyzer"
	@echo "  make summary   : 查看编译摘要"
	@echo "  make clean     : 清理生成文件"

################################################################################
# 目标定义
################################################################################
all: $(AIE_OUTPUT)
compile: $(AIE_OUTPUT)

#---------------------------------------------------------------------------#
# 编译 AIE
#---------------------------------------------------------------------------#
$(AIE_OUTPUT): $(MY_SOURCES)
	@mkdir -p $(BUILD_DIR)
	@( cd $(BUILD_DIR) && \
	    v++ --compile --config ../aie.cfg --mode aie --target=hw \
	        $(filter-out --aie.output-archive=$(AIE_OUTPUT),$(AIE_FLAGS)) \
	        --aie.output-archive=libadf.a \
	  2>&1 | tee -a ../log )
#---------------------------------------------------------------------------#
# AIE 仿真
#---------------------------------------------------------------------------#
sim: 
	@cp -r $(SRC_DIR)/data_input $(BUILD_DIR)
	@( cd $(BUILD_DIR) && \
	    aiesimulator --profile --dump-vcd=foo \
	  2>&1 | tee -a ../log )

sim1: 
	@cp -r $(SRC_DIR)/data_input $(BUILD_DIR)
	@( cd $(BUILD_DIR) && \
	    aiesimulator \
	  2>&1 | tee -a ../log )

#---------------------------------------------------------------------------#
# x86 仿真
#---------------------------------------------------------------------------#
x86sim:
	@mkdir -p $(BUILD_DIR)
	v++ --compile --config aie.cfg --mode=aie --target=x86sim $(AIE_FLAGS) 2>&1 | tee -a log && \
	x86simulator 2>&1 | tee -a log

#---------------------------------------------------------------------------#
# 其他工具快捷方式
#---------------------------------------------------------------------------#
profile: sim
trace:   sim
summary1:
	vitis_analyzer $(BUILD_DIR)/aiesimulator_output/default.aierun_summary
summary:
	vitis_analyzer $(BUILD_DIR)/Work/$(MY_APP).aiecompile_summary


################################################################################
# 清理
################################################################################
clean:
	rm -rf .Xil Work build.hw log log* aiesimulator_output* aiesimulator*.log \
	       x86simulator_output* *.xpe *.elf *.db *.soln Map_* xnw* *.lp *.json \
	       temp ISS_RPC_SERVER_PORT .crashReporter .AIE_SIM_CMD_LINE_OPTIONS \
	       system*.* trdata.aiesim vfs_work .wsdata _ide .ipynb_checkpoints \
	       vitis_analyzer* pl_sample_counts* pl_sample_count_*

a: clean all sim1
a1:  all sim1