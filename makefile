# TENNLab e-prop Implementation
#
# @file
# @version 0.1

CXX ?= g++
CC ?= gcc

FR_LIB = framework-open/lib/libframework.a
FR_INCLUDES = framework-open/include/
FR_CFLAGS = -std=c++11 -Wall -Wextra -O2 -march=native -Iinclude -Iframework-open/include -Iframework-open/include/utils -Ivendor/OpenCL-Wrapper/include $(CFLAGS)
FR_OBJ = framework-open/obj/framework.o framework-open/obj/processor_help.o framework-open/obj/properties.o

RISP_OBJ = framework-open/obj/risp.o framework-open/obj/risp_static.o
BPTT_OBJ = obj/shared.o obj/math_utils.o obj/data_utils.o obj/network_utils.o obj/forward_backward.o obj/optimizer.o obj/threading.o obj/csv.o obj/kernel.o

FRAMEWORK_DIR = framework-open/

all: framework-open/bin/network_tool \
	bin/bptt_learning

# Applications ################################################################
bin/bptt_learning: src/bptt_learning.cpp $(BPTT_OBJ) $(RISP_OBJ) $(FR_LIB)
	$(CXX) $(FR_CFLAGS) -o $@ $^

# Libraries ###################################################################
framework-open/lib/libframework.a: $(FR_OBJ) framework-open/include/framework.hpp
	ar r framework-open/lib/libframework.a $(FR_OBJ)
	ranlib framework-open/lib/libframework.a

# Objects #####################################################################
framework-open/obj/framework.o: framework-open/src/framework.cpp $(FR_INC)
	$(CXX) -c $(FR_CFLAGS) -o $@ framework-open/src/framework.cpp

framework-open/obj/processor_help.o: framework-open/src/processor_help.cpp $(FR_INC)
	$(CXX) -c $(FR_CFLAGS) -o framework-open/obj/processor_help.o framework-open/src/processor_help.cpp

framework-open/obj/properties.o: framework-open/src/properties.cpp $(FR_INC)
	$(CXX) -c $(FR_CFLAGS) -o framework-open/obj/properties.o framework-open/src/properties.cpp

framework-open/obj/risp.o: framework-open/src/risp.cpp $(FR_INC) $(RISP_INC)
	$(CXX) -c $(FR_CFLAGS) -o framework-open/obj/risp.o framework-open/src/risp.cpp

framework-open/obj/risp_static.o: framework-open/src/risp_static.cpp $(FR_INC) $(RISP_INC)
	$(CXX) -c $(FR_CFLAGS) -o framework-open/obj/risp_static.o framework-open/src/risp_static.cpp

obj/shared.o: src/shared.cpp 
	$(CXX) $(FR_CFLAGS) -o $@ -c $^

obj/math_utils.o: src/math_utils.cpp 
	$(CXX) $(FR_CFLAGS) -o $@ -c $^

obj/data_utils.o: src/data_utils.cpp 
	$(CXX) $(FR_CFLAGS) -o $@ -c $^

obj/network_utils.o: src/network_utils.cpp 
	$(CXX) $(FR_CFLAGS) -o $@ -c $^

obj/forward_backward.o: src/forward_backward.cpp 
	$(CXX) $(FR_CFLAGS) -o $@ -c $^

obj/optimizer.o: src/optimizer.cpp 
	$(CXX) $(FR_CFLAGS) -o $@ -c $^

obj/threading.o: src/threading.cpp 
	$(CXX) $(FR_CFLAGS) -o $@ -c $^

obj/csv.o: src/csv.cpp 
	$(CXX) $(FR_CFLAGS) -o $@ -c $^

obj/kernel.o: vendor/OpenCL-Wrapper/src/kernel.cpp
	$(CXX) $(FR_CFLAGS) -o $@ -c $^

# Utility ######################################################################
framework-open/bin/network_tool:
	( cd $(FRAMEWORK_DIR) && make bin/network_tool )

# Clean up #####################################################################
clean: clean_framework
	 rm -f bin/* obj/*

clean_framework:
	( cd framework-open/ ; make clean )

# end
