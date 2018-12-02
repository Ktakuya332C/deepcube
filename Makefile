CXX = g++
INC = -I include
FLG = -Wall -std=c++11

main: bin/trainer_sample_logloss bin/trainer_sample_squared

test: bin/test_cube bin/test_nn_math bin/test_nn_layer bin/test_nn_cost


bin/trainer_sample_logloss: build/trainer_sample_logloss.o build/cube.o build/nn_layer.o build/nn_math.o build/nn_cost.o
	$(CXX) $(FLG) $^ -o $@

bin/trainer_sample_squared: build/trainer_sample_squared.o build/cube.o build/nn_layer.o build/nn_math.o build/nn_cost.o
	$(CXX) $(FLG) $^ -o $@

bin/test_cube: build/test_cube.o build/cube.o
	$(CXX) $(FLG) $^ -o $@

bin/test_nn_math: build/test_nn_math.o build/nn_math.o
	$(CXX) $(FLG) $^ -o $@

bin/test_nn_layer: build/test_nn_layer.o build/nn_layer.o build/nn_math.o
	$(CXX) $(FLG) $^ -o $@

bin/test_nn_cost: build/test_nn_cost.o build/nn_cost.o
	$(CXX) $(FLG)  $^ -o $@


build/trainer_sample_logloss.o: src/trainer_sample_logloss.cc
	$(CXX) $(FLG) $(INC) -c $^ -o $@

build/trainer_sample_squared.o: src/trainer_sample_squared.cc
	$(CXX) $(FLG) $(INC) -c $^ -o $@

build/cube.o: src/cube.cc
	$(CXX) $(FLG) $(INC) -c $^ -o $@

build/nn_math.o: src/nn_math.cc
	$(CXX) $(FLG) $(INC) -c $^ -o $@

build/nn_layer.o: src/nn_layer.cc
	$(CXX) $(FLG) $(INC) -c $^ -o $@

build/nn_cost.o: src/nn_cost.cc
	$(CXX) $(FLG) $(INC) -c $^ -o $@

build/test_cube.o: test/test_cube.cc
	$(CXX) $(FLG) $(INC) -c $^ -o $@

build/test_nn_math.o: test/test_nn_math.cc
	$(CXX) $(FLG) $(INC) -c $^ -o $@

build/test_nn_layer.o: test/test_nn_layer.cc
	$(CXX) $(FLG) $(INC) -c $^ -o $@

build/test_nn_cost.o: test/test_nn_cost.cc
	$(CXX) $(FLG) $(INC) -c $^ -o $@


.PHONY: clean
clean:
	rm build/* bin/*

