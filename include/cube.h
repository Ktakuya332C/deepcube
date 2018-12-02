#ifndef CUBE_H_
#define CUBE_H_

const int state_size = 20 * 24;

enum Move {
  U_CC, U_CW, D_CC, D_CW,
  F_CC, F_CW, B_CC, B_CW,
  R_CC, R_CW, L_CC, L_CW,
};

class Cube {
public:
  Cube();
  ~Cube();
  void init();
  bool is_solved();
  void rotate(Move move);
  void get_state(bool* cur_state);
  void print_raw_state();
private:
  int* labels;
};

#endif // CUBE_H_
