#
# AMD AIE FFT Example Makefile
#
################################################################################
# Configurable variables
################################################################################
# AIE simulation only (no hardware image generation)
AIE_SIM_ONLY ?= false
# Enable FIFO depth checking
SIM_FIFO     ?= false

################################################################################
# Source files and directories
################################################################################
SRC_DIR   := src
MY_APP    := test_ssrfft_1m

# Include paths (including src/funcs and src/fam_kernels)
USER_INCLUDES := --include=$(CURDIR)/$(SRC_DIR) \
                 --include=$(CURDIR)/$(SRC_DIR)/funcs \
                 --include=$(CURDIR)/$(SRC_DIR)/fam_kernels \
				 --include=$(CURDIR)/$(SRC_DIR)/nn_kernels

# Main source file (absolute path)
MY_SOURCES := $(CURDIR)/$(SRC_DIR)/test_ssrfft_1m.cpp

################################################################################
# Output directories and files
################################################################################
BUILD_DIR := build.hw              # ← Output directory (same level as src)
AIE_OUTPUT := $(BUILD_DIR)/libadf.a

################################################################################
# Platform info
################################################################################
PLATFORM_USE := xilinx_vek280_base_202420_1
PLATFORM ?= $(PLATFORM_REPO_PATHS)/$(PLATFORM_USE)/$(PLATFORM_USE).xpfm

################################################################################
# Compilation options
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
# Common phony targets
################################################################################
.PHONY: help clean all compile sim sim1 x86sim profile trace analyze summary

help::
	@echo "Makefile Usage:"
	@echo "  make all       : Compile and generate libadf.a"
	@echo "  make sim       : Run AIE simulator (with profile + VCD)"
	@echo "  make sim1      : Run AIE simulator (functional only)"
	@echo "  make x86sim    : Run x86 functional simulator"
	@echo "  make profile   : Same as sim"
	@echo "  make trace     : Same as sim (includes VCD)"
	@echo "  make analyze   : Open Vitis Analyzer"
	@echo "  make summary   : View compilation summary"
	@echo "  make clean     : Clean up generated files"

################################################################################
# Target definitions
################################################################################
all: $(AIE_OUTPUT)
compile: $(AIE_OUTPUT)

#---------------------------------------------------------------------------#
# Compile AIE
#---------------------------------------------------------------------------#
$(AIE_OUTPUT): $(MY_SOURCES)
	@mkdir -p $(BUILD_DIR)
	@( cd $(BUILD_DIR) && \
	    v++ --compile --config ../aie.cfg --mode aie --target=hw \
	        $(filter-out --aie.output-archive=$(AIE_OUTPUT),$(AIE_FLAGS)) \
	        --aie.output-archive=libadf.a \
	  2>&1 | tee -a ../log )
#---------------------------------------------------------------------------#
# AIE simulation
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
# x86 simulation
#---------------------------------------------------------------------------#
x86sim:
	@mkdir -p $(BUILD_DIR)
	v++ --compile --config aie.cfg --mode=aie --target=x86sim $(AIE_FLAGS) 2>&1 | tee -a log && \
	x86simulator 2>&1 | tee -a log

#---------------------------------------------------------------------------#
# Tool shortcuts
#---------------------------------------------------------------------------#
profile: sim
trace:   sim
summary1:
	vitis_analyzer $(BUILD_DIR)/aiesimulator_output/default.aierun_summary
summary:
	vitis_analyzer $(BUILD_DIR)/Work/$(MY_APP).aiecompile_summary

################################################################################
# Cleanup
################################################################################
clean:
	rm -rf .Xil Work build.hw log log* aiesimulator_output* aiesimulator*.log \
	       x86simulator_output* *.xpe *.elf *.db *.soln Map_* xnw* *.lp *.json \
	       temp ISS_RPC_SERVER_PORT .crashReporter .AIE_SIM_CMD_LINE_OPTIONS \
	       system*.* trdata.aiesim vfs_work .wsdata _ide .ipynb_checkpoints \
	       vitis_analyzer* pl_sample_counts* pl_sample_count_*

a: clean all sim1
a1:  all sim1
