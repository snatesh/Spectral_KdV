#ifndef AUXILIAR_H
#define AUXILIAR_H
#include<Eigen/Dense>
#include<vector>
#include<memory>

// templated function ptr for f:T->T
// lambdas can be passed as well, but without capture params (per c++11 standard)
template<typename T>
using funcPtr = T (*) (T);

template<typename T>
using funcPtrRef = T (*) (T&);

// templated shared ptr
template<typename T>
using smartPtr = std::shared_ptr<T>;

// templated dynamic Eigen vector
template<typename T>
using VectorX = Eigen::Matrix<T,Eigen::Dynamic,1>;

// templated eigen dynamic matrix
template<typename T>
using MatrixX = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;

// templated vector of matrices (order 3 tensor)
template<typename T>
using TensorX3 = std::vector<MatrixX<T>>; 

// returns func(t0:dt:tf)
template<typename T>
VectorX<T> range(T t0, T tf, T dt, funcPtr<T> func = nullptr);

#include "../src/auxiliar.C"

#endif
