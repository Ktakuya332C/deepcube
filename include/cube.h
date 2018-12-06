#ifndef CUBE_H_
#define CUBE_H_

const int n_move = 12;
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
  void restore(Cube const &other);
  bool is_solved();
  bool is_solved_hypo(Move move);
  void rotate(Move move);
  Move rotate_random();
  void get_state(double* cur_state);
  void get_state_hypo(Move move, double* next_state);
  void print_raw_state();
private:
  int* labels;
};

#endif // CUBE_H_
