BOOST_DIR=/home/ampereira/tools/boost/1.61.0-icc

################################################################################
# by André Pereira (LIP-Minho)
################################################################################

SHELL = /bin/sh

DEFINES =

LIB_NAME = libPrngManager

ifeq ($(PRNGM_INTEL),yes)
CXX = icpc  -DD_HEPF_INTEL -qopenmp
LD  = icpc  -DD_HEPF_INTEL -qopenmp
else
CXX = g++
LD  = g++
endif

SUFFIX=lib

#-Wno-unused-parameter   -Wno-maybe-uninitialized -Wno-return-type -Wno-unused-but-set-parameter  -Wno-sign-compare -Wno-deprecated-declarations

CXXFLAGS   = -DD_VERBOSE -Wall -Wextra -std=c++0x -DD_MULTICORE -DD_REPORT -lboost_thread -Wno-unused-parameter -Wno-uninitialized -Wno-unused-variable -Wno-missing-field-initializers -Wno-unused-but-set-parameter 
INCLUDES = -I/src/ 

KNC_FLAGS= -Wall

ifeq ($(PRNGM_INTEL),yes)
	CXXFLAGS += -DD_MKL
	OTHER_LIBS += -mkl=parallel -lmkl_blas95_lp64 -lmkl_lapack95_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core 
endif

DIR=$(shell pwd)
DIR=${$DIR%$SUFFIX}


ifeq ($(PRNGM_ROOT),yes)
	ROOTCFLAGS = $(shell root-config --cflags)
	ROOTGLIBSAUX  = $(shell root-config --glibs)
	GPU_ROOTGLIBS= --compiler-options " $(subst -limf,, $(ROOTGLIBSAUX)) "
	ROOTGLIBS=$(subst -limf,, $(ROOTGLIBSAUX))
	CXXFLAGS += -DD_ROOT $(ROOTCFLAGS)
	INCLUDES += -I$(ROOTSYS)/include
endif

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CXXFLAGS+= -Wno-unused-command-line-argument
endif

ifeq ($(PRNGM_KNC),yes)
	KNC_FLAGS += -DD_KNC -I${MKLROOT}/include -offload-option,mic,compiler," -L${MKLROOT}/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core"
endif

ifeq ($(PRNGM_MPI),yes)
	ifeq ($(PRNGM_INTEL),yes)
		CXX = mpiicpc -DD_HEPF_INTEL
		LD  = mpiicpc -DD_HEPF_INTEL
		CXXFLAGS += -DD_MPI
	else
		CXX = mpicxx
		LD  = mpicxx
		CXXFLAGS += -DD_MPI
	endif
endif


ifeq ($(PRNGM_DEBUG),yes)
	CXXFLAGS += -g
else
	CXXFLAGS += -O3
endif

ifeq ($(strip $(BOOST_DIR)),)
	# GPU OFFLOAD
	ifeq ($(PRNGM_GPU),yes)
		KNC_FLAGS=-DD_LOL
		ifeq ($(PRNGM_MPI),yes)
			INCLUDES = -I$(ROOTSYS)/include
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
	INCLUDES += -I$(BOOST_DIR)/include
	LIBS = -L$(BOOST_DIR)/lib
	GPULIBS = -L$(BOOST_DIR)/lib

	# 	# GPU OFFLOAD
	
	ifeq ($(PRNGM_GPU),yes)
		KNC_FLAGS=-DD_LOL
		ifeq ($(PRNGM_MPI),yes)
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
SRC_DIR = src
LIB_DIR = lib
BUILD_DIR = build
SRC = $(wildcard $(SRC_DIR)/*.cxx)
OBJ = $(patsubst src/%.cxx,build/%.o,$(SRC))


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
