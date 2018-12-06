#include "cube.h"
#include <map>
#include <random>
#include <iostream>

namespace {

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(0, n_move-1);

const int u_cycles[5][4] = {
  { 0,  1,  3,  2},
  {24, 26, 27, 25},
  {13, 11,  9, 22},
  {38, 37, 36, 47},
  {12, 10,  8, 23},
};

const int d_cycles[5][4] = {
  { 4,  5,  7,  6},
  {30, 32, 33, 31},
  {14, 16, 18, 21},
  {43, 44, 45, 46},
  {15, 17, 19, 20},
};

const int f_cycles[5][4] = {
  {10, 11, 17, 16},
  {37, 29, 44, 28},
  { 2, 12,  5, 15},
  {27, 41, 30, 40},
  { 3, 18,  4,  9},
};

const int b_cycles[5][4] = {
  {20, 21, 23, 22},
  {46, 35, 47, 34},
  { 6, 19,  1,  8},
  {33, 42, 24, 39},
  { 7, 13,  0, 14},
};

const int r_cycles[5][4] = {
  {12, 13, 19, 18},
  {38, 42, 45, 41},
  {11,  1, 21,  5},
  {29, 26, 35, 32},
  {17,  3,  23,  7},
};

const int l_cycles[5][4] = {
  { 8,  9, 15, 14},
  {36, 40, 43, 39},
  {10,  4, 20,  0},
  {28, 31, 34, 25},
  {16,  6, 22,  2},
};

const std::map<Move, Move> counter_move = {
  {U_CC, U_CW},
  {U_CW, U_CC},
  {D_CC, D_CW},
  {D_CW, D_CC},
  {F_CC, F_CW},
  {F_CW, F_CC},
  {B_CC, B_CW},
  {B_CW, B_CC},
  {R_CC, R_CW},
  {R_CW, R_CC},
  {L_CC, L_CW},
  {L_CW, L_CC},
};

void rotate_forward(int* labels, const int cycles[5][4]) {
  for (int i=0; i<5; i++) {
    int tmp = labels[cycles[i][3]];
    labels[cycles[i][3]] = labels[cycles[i][2]];
    labels[cycles[i][2]] = labels[cycles[i][1]];
    labels[cycles[i][1]] = labels[cycles[i][0]];
    labels[cycles[i][0]] = tmp;
  }
}

void rotate_backward(int* labels, const int cycles[5][4]) {
  for (int i=0; i<5; i++) {
    int tmp = labels[cycles[i][0]];
    labels[cycles[i][0]] = labels[cycles[i][1]];
    labels[cycles[i][1]] = labels[cycles[i][2]];
    labels[cycles[i][2]] = labels[cycles[i][3]];
    labels[cycles[i][3]] = tmp;
  }
}

}

Cube::Cube() {
  labels = new int[48];
  init();
}

Cube::~Cube() {
  delete[] labels;
}

void Cube::init() {
  for (int i=0; i<48; i++) {
    labels[i] = i;
  }
}

void Cube::restore(Cube const &other) {
  for (int i=0; i<48; i++) {
    labels[i] = other.labels[i];
  }
}

bool Cube::is_solved() {
  for (int i=0; i<48; i++) {
    if (labels[i] != i) {
      return false;
    }
  }
  return true;
}

bool Cube::is_solved_hypo(Move move) {
  rotate(move);
  bool flag = is_solved();
  rotate(counter_move.at(move));
  return flag;
}

void Cube::get_state(double* cur_state) {
  for (int i=0; i<20*24; i++) {
    cur_state[i] = 0;
  }
  for (int i=0; i<8; i++) {
    cur_state[i*24 + labels[i]] = 1;
  }
  for (int i=8; i<20; i++) {
    cur_state[i*24 + labels[i+16] - 24] = 1;
  }
}

void Cube::get_state_hypo(Move move, double* next_state) {
  rotate(move);
  get_state(next_state);
  rotate(counter_move.at(move));
}

void Cube::print_raw_state() {
  for (int i=0; i<48; i++) {
    /* std::cout << "Position: " << i;
    std::cout << " Value: " << labels[i] << std::endl; */
    std::cout << labels[i] << " ";
  }
  std::cout << std::endl;
}

void Cube::rotate(Move move) {
  switch(move) {
    case U_CW: rotate_forward(labels, u_cycles); break;
    case U_CC: rotate_backward(labels, u_cycles); break;
    case D_CW: rotate_forward(labels, d_cycles); break;
    case D_CC: rotate_backward(labels, d_cycles); break;
    case F_CW: rotate_forward(labels, f_cycles); break;
    case F_CC: rotate_backward(labels, f_cycles); break;
    case B_CW: rotate_forward(labels, b_cycles); break;
    case B_CC: rotate_backward(labels, b_cycles); break;
    case R_CW: rotate_forward(labels, r_cycles); break;
    case R_CC: rotate_backward(labels, r_cycles); break;
    case L_CW: rotate_forward(labels, l_cycles); break;
    case L_CC: rotate_backward(labels, l_cycles); break;
  }
}

Move Cube::rotate_random() {
  Move move = static_cast<Move>(dis(gen));
  rotate(move);
  return move;
}
