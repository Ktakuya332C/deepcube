CXX = g++
INC = -I include -I /usr/local/opt/openblas/include
FLG = -Wall -O2 -std=c++11
LIB = -lopenblas -L /usr/local/opt/openblas/lib

SRCS = $(shell find src -type f -name *.cc)
OBJS = $(patsubst src/%.cc, build/src/%.o, $(SRCS))

MAINSRCS = $(shell find exec -type f -name *.cc)
MAINOBJS = $(patsubst exec/%.cc, build/exec/%.o, $(MAINSRCS))
MAINTGTS = $(patsubst exec/%.cc, bin/exec/%, $(MAINSRCS))

TESTSRCS = $(shell find test -type f -name *.cc)
TESTOBJS = $(patsubst test/%.cc, build/test/%.o, $(TESTSRCS))
TESTTGTS = $(patsubst test/%.cc, bin/test/%, $(TESTSRCS))

.PHONY: main
main: $(MAINTGTS)

.PHONY: test
test: $(TESTTGTS)
	$(foreach prog, $(TESTTGTS), $(prog) &&) true

.PHONY: clean
clean:
	rm -rf build/* bin/*

bin/exec/%: build/exec/%.o $(OBJS)
	@mkdir -p bin/exec;
	$(CXX) $(FLG) $^ -o $@ $(LIB)

bin/test/%: build/test/%.o $(OBJS)
	@mkdir -p bin/test;
	$(CXX) $(FLG) $^ -o $@ $(LIB)

build/exec/%.o: exec/%.cc
	@mkdir -p build/exec;
	$(CXX) $(FLG) $(INC) -c $^ -o $@

build/test/%.o: test/%.cc
	@mkdir -p build/test;
	$(CXX) $(FLG) $(INC) -c $^ -o $@

build/src/%.o: src/%.cc
	@mkdir -p build/src;
	$(CXX) $(FLG) $(INC) -c $^ -o $@

