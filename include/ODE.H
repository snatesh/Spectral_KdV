#ifndef ODE_H 
#define ODE_H

/* ODE is a general class to solve ODEs of the form Du/dt = Lu + N(u), 
 * where L is a linear operator and B can be nonlinear. Users can select from
 * either the AB2BD2 semi-implicit scheme, or the ETDRK2 exponential
 * time differencing rk2 scheme. 
 * TODO: add support for
 *			- richardson extrap for initializing u0, u1 and u2
 *			-	2, 3 space dim  
 *			-	explicit time dependence in nonlin term
 *			-	ctor for funcPtr u0 (maybe convenient sometimes)
 *			- specifying linear solver (currently uses column pivoted householder qr for generality) 
*/

#include<Eigen/Dense>
#include<Eigen/StdVector>
#include<auxiliar.H>

template<typename T,typename U>
class ODE
{
	public:
		// construct with linear part, nonlinear funcPtr, initial cond, timespan
		ODE(MatrixX<T>& _L, funcPtrRef<MatrixX<T>> _N, 
					 MatrixX<T>& u0, VectorX<U>& _x, VectorX<U>& _T);  
		// construct with 0 nonlinear part
		ODE(MatrixX<T>& _L, MatrixX<T>& u0, VectorX<U>& _x, VectorX<U>& _T);  
		// construct with 0 linear part
		ODE(funcPtrRef<MatrixX<T>>_N, MatrixX<T>& u0, VectorX<U>& _x, VectorX<U>& _T);

		// static method to assemble system and for RAII design for later abstraction
		static smartPtr<ODE<T,U>> Assemble(MatrixX<T>& _L, funcPtrRef<MatrixX<T>> _N, 
								 													MatrixX<T>& u0, VectorX<U>& _x, VectorX<U>& _T);

		static smartPtr<ODE<T,U>> Assemble(MatrixX<T>& _L, MatrixX<T>& u0, 
																					VectorX<U>& _x, VectorX<U>& _T);
		// solve vi ODE method and householderQr with pivoting to solve the
		// linear system resulting from isolating the explict term 
		void solve(const std::string& solver);
		TensorX3<T> getSol() { return sol;}	

		void ab2bd2();
		void etdrk2();

	private:
		// lin op
		MatrixX<T> L;
		// nonlin func
		funcPtrRef<MatrixX<T>> Nfunc;
		// timespan
		VectorX<U> t;
		// grid 
		VectorX<U> x;
		// solution evaluated over T
		TensorX3<T> sol;
		// nonlin or lin
		bool nonlin; 
		bool lin;
		ODE(); // privatize default ctor, i.e remove from outer scope
};

#include "../src/ODE.C"
#endif
