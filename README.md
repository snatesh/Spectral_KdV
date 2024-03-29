# Spectral_KdV
General, templated implementation of an order 2 semi-implicit  Adams Bashforth/backward‐differentiation time stepping scheme and an order 2 exponential Runge-Kutta method applied to a spectral discretization in space of the Korteweg–de Vries equation with smooth initial data and periodic boundary conditions. 

The implementation is built on top of Eigen and Eigen's wrapper around FFTW. A C++ binding to Python's matplotlib API is also included for visualization purposes.

The methods and application are summarized in the pdf document. Below is a short animation depicting two solitons passing through each other, after which they recover their original shapes.

![Sample Frame](soliton_passthru.gif)
