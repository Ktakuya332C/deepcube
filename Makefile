CXX = g++
INC = -I include
FLG = -Wall -std=c++11

main: bin/main

test: bin/test_cube

bin/main: build/main.o build/cube.o
	$(CXX) $(FLG) $^ -o $@

bin/test_cube: build/test_cube.o build/cube.o
	$(CXX) $(FLG) $^ -o $@

build/main.o: src/main.cc
	$(CXX) $(FLG) $(INC) -c $^ -o $@

build/cube.o: src/cube.cc
	$(CXX) $(FLG) $(INC) -c $^ -o $@

build/test_cube.o: test/test_cube.cc
	$(CXX) $(FLG) $(INC) -c $^ -o $@

clean:
	rm build/* bin/*

