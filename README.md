# A Continuous-time Perspective for Modeling Acceleration in Riemannian Optimization
Code to replicate experiments in https://arxiv.org/abs/1910.10782

For any questions, please contact me at antonio.orvieto@inf.ethz.ch

# Requirements
All scripts are written in Matlab 2017b, but should work for any recent version of Matlab. 

No libraries (e.g. Manopt) are required: geometry is implemented directly.

# Contents of this repo

We test the performance of SIRNAG, a semi-implicit discretization of RNAG-ODE, with the following experiments:

 - `distance_minimization_toy.m` minimizes the distance to a point on manifolds of any costant positive or negative curvature. Used to generate Figure 1.

 - `sphere_eigproblem.m` minimizes the Rayleigh quotient on the sphere. Used to generate Figure 2.
 
 - `convergence_trajectory_SIRNAG.m` tests the convergence of SIRNAG to RNAG-ODE. Used to generate Figure 3.
