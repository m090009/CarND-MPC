#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// TODO: Set the timestep length and duration
const size_t N = 12;
const double dt = 0.05;


// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// Reference values for errors and velocity
// 80 to get the speed up to 70
double ref_v = 85;

// The solver takes a single vector for both the state and actuator values 
// so here we identify the w starse 
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) {
    this->coeffs = coeffs;
  }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // TODO: implement MPC
    // `fg` a vector of the cost constraints, `vars` is a vector   double prev_a{0};
    // NOTE: You'll probably go back and forth between this function and
    // the Solver function below.
    
    // init cost sum to 0
    fg[0] = 0;

    // The part of the cost based on the reference state.
    for (int t = 0; t < N; t++) {
      fg[0] += CppAD::pow(vars[cte_start + t], 2);
      fg[0] += CppAD::pow(vars[epsi_start + t], 2);
      fg[0] += CppAD::pow(vars[v_start + t] - ref_v, 2);
    }

    // Minimize the use of actuators which will constrain erratic control inputs
    // for making smooth turns
    for (int t = 0; t < N - 1; t++) {
      fg[0] += CppAD::pow(vars[delta_start + t], 2);
      // Weight for
      fg[0] += 10 * CppAD::pow(vars[a_start + t], 2);

    }


    // Minimize the value gap between sequential actuations to have a 
    // more consistent control decisions .

    for (int t = 0; t < N - 2; t++) {
      // Added a 100 and 10 to get a smoother transition from -25 to 25 degrees
      // Steering weight
      fg[0] += 700 * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
      fg[0] += CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
    }


    // Init FG, here we increase each of the indecies by 1 
    // because fg[0] is reserved for the sum of the errors
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];



    // We add 1
    for (int t = 0; t < N - 1; t++) {
      // The state at time t + 1 + 1 to mind the delay .
      AD<double> x1 = vars[x_start + t + 1];
      AD<double> y1 = vars[y_start + t + 1];
      AD<double> psi1 = vars[psi_start + t + 1];
      AD<double> v1 = vars[v_start + t + 1];
      AD<double> cte1 = vars[cte_start + t + 1];
      AD<double> epsi1 = vars[epsi_start + t + 1];

      // The state at time t.
      AD<double> x0 = vars[x_start + t];
      AD<double> y0 = vars[y_start + t];
      AD<double> psi0 = vars[psi_start + t];
      AD<double> v0 = vars[v_start + t];
      AD<double> cte0 = vars[cte_start + t];
      AD<double> epsi0 = vars[epsi_start + t];

      // Only consider the actuation at time t.
      AD<double> delta0 = vars[delta_start + t];
      AD<double> a0 = vars[a_start + t];

      // 3 dimensions
      AD<double> f0 = coeffs[0] + coeffs[1] * x0 + coeffs[2]*x0*x0 + coeffs[3]*x0*x0*x0;
      AD<double> psides0 = CppAD::atan(coeffs[1]+2*coeffs[2]*x0 + 3 * coeffs[3]*x0*x0);


      // Recall the equations for the model:
      // x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
      // y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
      // psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
      // v_[t+1] = v[t] + a[t] * dt
      // cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
      // epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt
      // + 2 to mind the delay time steps
      fg[2 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[2 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[2 + psi_start + t] = psi1 - (psi0 + v0 * delta0 / Lf * dt);
      fg[2 + v_start + t] = v1 - (v0 + a0 * dt);
      fg[2 + cte_start + t] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
      fg[2 + epsi_start + t] = epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt);
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

Result MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;


  double x = state[0];
  double y = state[1];
  double psi = state[2];
  double v = state[3];
  double cte = state[4];
  double epsi = state[5];

  // TODO: Set the number of model variables (includes both states and inputs).
  // For example: If the state is a 4 element vector, the actuators is a 2
  // element vector and there are 10 timesteps. The number of variables is:
  //
  // 4 * 10 + 2 * 9
  size_t n_vars = N * 6 + (N - 1) *  2;
  // TODO: Set the number of constraints
  size_t n_constraints = N * 6;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0.0;
  }

  // Set the initial variable values
  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[epsi_start] = epsi;

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
    
  // Set all non-actuators upper and lowerlimits to anything
  for (int i = 0; i < delta_start; i++) {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }

  // The upper and lower limits of delta are set to -25 and 25
  // degrees (values in radians).
  // NOTE: Feel free to change this to something else.
  // double deg2rad(double x) { return x * pi() / 180; };
  // 0.436332 is 25 in rads
  for (int i = delta_start; i < a_start; i++) {
    vars_lowerbound[i] = - 0.436332;
    vars_upperbound[i] = 0.436332;
  }
  // cout << "Previous delta :" << prev_delta << endl;
  // cout << "Previous a :" << prev_a << endl;
  // Constraint acctuator values to the previous values delta
  for (int i = delta_start; i < delta_start + latency_timestep; i++){
    vars_lowerbound[i] = prev_delta;
    vars_upperbound[i] = prev_delta;
  }


  // Acceleration/decceleration upper and lower limits.
  // NOTE: Feel free to change this to something else.
  for (int i = a_start; i < n_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }
 // Constraint acctuator values to the previous values acceleration
  for (int i = a_start; i < a_start + latency_timestep; i++) {
    vars_lowerbound[i] = prev_a;
    vars_upperbound[i] = prev_a;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  // Set constraints values to 0 
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }
  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;

//  vector<double> prev_actuator_values = {prev_delta, prev_a};
  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  // std::cout << "Cost " << cost << std::endl;

  // TODO: Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.
  // vector<double> result;
  // result.push_back(solution.x[delta_start]);
  // result.push_back(solution.x[a_start]);
  // Resut struct to hold the values ofr the optimization
  Result result;

  for (auto i = 0; i < N - 1; i++) {
    result.x.push_back(solution.x[x_start + i]);
    result.y.push_back(solution.x[y_start + i]);
    result.delta.push_back(solution.x[delta_start + i]);
    result.a.push_back(solution.x[a_start + i]);
  }

  return result;
}

