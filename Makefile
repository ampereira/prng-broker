BOOST_DIR=/home/ampereira/tools/boost/1.61.0-icc

################################################################################
# Makefile for New Physics Analysis
#
# by André Pereira (LIP-Minho)
################################################################################

SHELL = /bin/sh

DEFINES =

LIB_NAME = libHEPFrame

ifeq ($(HEPF_INTEL),yes)
#CXX        = icpc  -DD_HEPF_INTEL -xMIC-AVX512
#LD         = icpc  -DD_HEPF_INTEL -xMIC-AVX512
CXX        = icpc  -DD_HEPF_INTEL -qopenmp
LD         = icpc  -DD_HEPF_INTEL -qopenmp
else
CXX        = g++
LD         = g++
endif

SUFFIX=lib

ROOTCFLAGS = $(shell root-config --cflags)
ROOTGLIBSAUX  = $(shell root-config --glibs)
GPU_ROOTGLIBS= --compiler-options " $(subst -limf,, $(ROOTGLIBSAUX)) "
KNC_FLAGS= -Wall
OTHER_LIBS =  -lboost_program_options
ROOTGLIBS=$(subst -limf,, $(ROOTGLIBSAUX))

ifeq ($(HEPF_INTEL),yes)
	OTHER_LIBS += -mkl=parallel -lmkl_blas95_lp64 -lmkl_lapack95_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core 
endif

DIR=$(shell pwd)
DIR=${$DIR%$SUFFIX}

# With delphes inputs
#CXXFLAGS   = -fopenmp -Wall -Wextra -std=c++11 -DD_MULTICORE -DD_REPORT -Wno-deprecated-declarations -DD_DELPHES $(ROOTCFLAGS)

# Without delphes inputs
CXXFLAGS   = -DD_ROOT -DD_THREAD_BALANCE -DD_VERBOSE -Wno-unused-parameter -Wno-unused-variable -Wno-uninitialized -Wno-maybe-uninitialized -Wno-return-type -Wno-unused-but-set-parameter -Wno-missing-field-initializers -Wno-sign-compare -Wall -Wextra -std=c++0x -DD_MULTICORE -DD_REPORT -Wno-deprecated-declarations $(ROOTCFLAGS) -lboost_thread 


UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CXXFLAGS+= -Wno-unused-command-line-argument
endif

ifeq ($(HEPF_KNC),yes)
#	KNC_FLAGS += -DD_KNC -qoffload-attribute-target=mic -qopenmp
	KNC_FLAGS += -DD_KNC -I${MKLROOT}/include -offload-option,mic,compiler," -L${MKLROOT}/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core"
endif

ifeq ($(HEPF_MPI),yes)
	ifeq ($(HEPF_INTEL),yes)
		CXX = mpiicpc -DD_HEPF_INTEL #-xMIC-AVX512
		LD  = mpiicpc -DD_HEPF_INTEL #-xMIC-AVX512
		CXXFLAGS += -DD_MPI
	else
		CXX = mpicxx
		LD  = mpicxx
		CXXFLAGS += -DD_MPI
	endif
endif

ifeq ($(HEPF_THREAD_BALANCE),no)
	CXXFLAGS:=$(filter-out -DD_THREAD_BALANCE,$(CXXFLAGS))
endif

ifeq ($(HEPF_ROOT),no)
	CXXFLAGS:=$(filter-out -DD_ROOT,$(CXXFLAGS))
	ROOTGLIBS = -DD_NO_ROOT
	GPU_ROOTGLIBS = -DD_NO_ROOT
endif

ifeq ($(HEPF_SCHEDULER),yes)
	CXXFLAGS += -DD_SCHEDULER
endif

ifeq ($(HEPF_AFFINITY),yes)
	CXXFLAGS += -DD_AFFINITY
endif

#ifeq ($(INTEL),yes)
#	CXX = icpc
#	LD = icpc
#	CXXFLAGS   = -fopenmp -Wall -Wextra -std=c++11 -DD_MULTICORE -DD_REPORT $(ROOTCFLAGS) -lboost_thread
#endif

ifeq ($(HEPF_DEBUG),yes)
	CXXFLAGS += -g
else
	CXXFLAGS += -O3
endif

ifeq ($(HEPF_TRAND),yes)
	CXXFLAGS += -DD_TRAND
endif

ifeq ($(strip $(BOOST_DIR)),)
	INCLUDES = -I$(ROOTSYS)/include -I../tools/ -I../tools/Delphes-3.2.0/


	# GPU OFFLOAD
	ifeq ($(HEPF_GPU),yes)
		KNC_FLAGS=-DD_LOL
		ifeq ($(HEPF_MPI),yes)
			INCLUDES = -I$(ROOTSYS)/include -I../tools/ -I../tools/Delphes-3.2.0/
			CXX = nvcc $(INCLUDES) $(GPULIBS) -c -O3 -lcurand -ccbin=mpiicpc --compiler-options "
			LD = nvcc $(INCLUDES) $(GPULIBS) -O3 -lcurand -ccbin=mpiicpc --compiler-options "
			CXXFLAGS += -DD_GPU -DD_HEPF_INTEL -DD_MPI
			LIBS += "
			ROOTGLIBS = $(GPU_ROOTGLIBS)
		else
			INCLUDES = -I$(ROOTSYS)/include -I../tools/ -I../tools/Delphes-3.2.0/
			CXX = nvcc $(INCLUDES) -c -O3 -lcurand --compiler-options "
			LD = nvcc $(INCLUDES) -O3 -lcurand --compiler-options "
			CXXFLAGS += -DD_GPU
			LIBS += "
			ROOTGLIBS = $(GPU_ROOTGLIBS)
		endif
	endif
else
	INCLUDES = -I$(ROOTSYS)/include -I$(BOOST_DIR)/include -I../tools/ -I../tools/Delphes-3.2.0/
	LIBS = -L$(BOOST_DIR)/lib
	GPULIBS = -L$(BOOST_DIR)/lib

# 	# GPU OFFLOAD
	
	ifeq ($(HEPF_GPU),yes)
		KNC_FLAGS=-DD_LOL
		ifeq ($(HEPF_MPI),yes)
			CXX = nvcc $(INCLUDES) $(GPULIBS) -c -O3 -lcurand -ccbin=mpiicpc --compiler-options "
			LD = nvcc $(INCLUDES) $(GPULIBS) -O3 -lcurand -ccbin=mpiicpc --compiler-options "
			CXXFLAGS += -DD_GPU -DD_HEPF_INTEL -DD_MPI
			LIBS += "
			ROOTGLIBS = $(GPU_ROOTGLIBS)
		else
			CXX = nvcc $(INCLUDES) -c -O3 -lcurand --compiler-options "
			LD = nvcc $(INCLUDES) -O3 -lcurand --compiler-options "
			CXXFLAGS += -DD_GPU
			LIBS += "
			ROOTGLIBS = $(GPU_ROOTGLIBS)
		endif
	endif
endif

################################################################################
# Control awesome stuff
################################################################################
#OFFLOAD_MIC_DIR=../Analysis/TTHoffload/offload/mic
# OFFLOAD_GPU_DIR=../Analysis/TTHoffload/offload/gpu
SRC_DIR = src
LIB_DIR = lib
BUILD_DIR = build
SRC = $(wildcard $(SRC_DIR)/*.cxx)
OBJ = $(patsubst src/%.cxx,build/%.o,$(SRC))
#OFFLADO_MIC_OBJ = $(wildcard $(OFFLOAD_MIC_DIR)/build/*.o)
# OFFLADO_GPU_OBJ = $(wildcard $(OFFLOAD_GPU_DIR)/build/*.o)


DEPS = $(patsubst build/%.o,build/%.d,$(OBJ))
LIB = $(addsufix .a, $(LIB_NAME))

vpath %.cxx $(SRC_DIR)

################################################################################
# Rules
################################################################################

.DEFAULT_GOAL = all

#$(BUILD_DIR)/PseudoRandomGeneratorKNC.o: PseudoRandomGeneratorKNC.cxx
#	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $(KNC_FLAGS) $(LIBS) $(OTHER_LIBS) $< -o $@ $(LIBS) $(OTHER_LIBS)

$(BUILD_DIR)/%.o: %.cxx
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $(KNC_FLAGS) $(LIBS) $(ROOTGLIBS) $(OTHER_LIBS) $< -o $@ $(ROOTGLIBS) $(OTHER_LIBS) $(KNC_FLAGS)

$(LIB_DIR)/$(LIB_NAME).a: $(OBJ)
	ar -r $@ $(OBJ)

$(BUILD_DIR)/$(LIB_NAME): $(OBJ)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(KNC_FLAGS) $(LIBS) $(ROOTGLIBS) $(OTHER_LIBS) -o $@ $(OBJ) $(ROOTGLIBS) $(OTHER_LIBS) $(KNC_FLAGS)
	
checkdirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(LIB_DIR)

#all: checkdirs $(BUILD_DIR)/PseudoRandomGeneratorKNC.o $(LIB_DIR)/$(LIB_NAME).a
all: checkdirs $(LIB_DIR)/$(LIB_NAME).a

test: checkdirs $(BUILD_DIR)/$(LIB_NAME)

clean:
	rm -f $(BUILD_DIR)/* $(LIB_DIR)/*
