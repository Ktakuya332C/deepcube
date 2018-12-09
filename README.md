DeepCube
----
One of Rubik's Cube solvers with Deep Reinforcement Learning.
The algorithm is based on the following paper.
* S. McAleer et al. (2018), Solving the Rubik's Cube without Human Knowledge

## Disclaimer
* The author of this repository does not have any relationships with the author of the paper cited above.
* This implements an algorithm a little different from the paper. The differences are summarised in the section below.
* The authors of the paper have already published an [interactive website](http://deepcube.igb.uci.edu/) that shows how the algorithm solves the Rubik's Cube.

## Usage
First, install [OpenBLAS](https://www.openblas.net/)
```
MacOS
$ brew install openblas
Ubuntu
$ sudo apt-get install libopenblas-dev
```
Then, get this repository and make
```
$ git clone https://github.com/Ktakuya332C/deepcube
$ cd deepcube
$ make
```
Solve randomly scrambled Rubik's Cubes by
```
$ bin/exec/search_cube
Scramble the cube. The order of moves is
D' R' U F' L' L D R' L' R'
Solved the cube. The order of moves is
R' L R' D' F U' R D
```
Paste these seqeuences to a [website](https://rubiks-cu.be/#cubesolver) to see how the algorithm solved the cube.

## Training a neural network
Train a neural network employed in the algorithm by
```
$ bin/exec/trainer_cube
```
which will train a network and save its parameters under `/tmp/`. 

Since the executable `search_cube` uses parameters in `data/` directory by default, to use the trained parameters, copy files like `/tmp/dense_layer_*` to `data/`. Then, the next execution of `seach_cube`
```
$ bin/exec/search_cube
```
will use the trained parameters.

## Difference
* The size of the neural network used in this implementation is small compared to the size of the network reported in the original paper. As a result of this size reduction, the performance of this implementation does not match the one reported in the original paper.
* The nonlinearity of the neural network is ReLU in this implementation despite the fact that the original paper uses elu.
* Originally, the optimization algorithm of the neural network is RMSProp. Here this implementation uses [Rprop](http://www.inf.fu-berlin.de/lehre/WS06/Musterererkennung/Paper/rprop.pdf) because of its simplicity.
* The interpretation of the Rubik's Cube as a Markov decision process is different from the original paper. The original paper describes it as a continuing task without an end state, meaning the task continues even if the Rubik's Cube is solved. This leads to the training algorithm described in Algorithm 1 of the original paper, which does not separate the solved state as a special case.
However, this implementation separates the solved state as a special case because we interpret the solved state as an end state of the MDP that terminates the task. This leads to an algorithm like [this](https://github.com/Ktakuya332C/deepcube/blob/master/exec/trainer_cube.cc#L60-L64).

## ToDo
* Refactoring ...

