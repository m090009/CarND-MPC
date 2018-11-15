#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

using namespace std;
// Resut struct to hold the values ofr the optimization
struct Result {
  vector<double> x;
  vector<double> y;
  vector<double> a;
  vector<double> delta;
};

class MPC {
 public:
  MPC();

  virtual ~MPC();

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuations.
  Result Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);
  // Previous a and delta values initialized
  double prev_a{0};
  double prev_delta{.1};
  // 100 ms latency which count for 2 time steps with out current dt 0.05
  const int latency_timestep = 2;
};

#endif /* MPC_H */
