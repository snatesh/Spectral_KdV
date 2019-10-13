#include<ODE.H>
#include<complex>
#include<iostream>
#include<matplotlibcpp.h>

namespace plt = matplotlibcpp;

// construct with linear part, nonlinear funcPtr, initial cond,
template<typename T,typename U>
ODE<T,U>::ODE(MatrixX<T>& _L, funcPtrRef<MatrixX<T>> _N, MatrixX<T>& u0, VectorX<U>& _x, VectorX<U>& _T)
{
	if (!std::is_floating_point<U>::value || 
			!std::is_floating_point<T>::value &&
			!(std::is_same<T, std::complex<double>>::value ||
				std::is_same<T, std::complex<float>>::value ||
				std::is_same<T,std::complex<long double>>::value))
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " expects floating point type" << std::endl;
		exit(1);
	}

	// load L and grid
	L = _L;
	x = _x;
	// load funcPtr
	Nfunc = _N;
	// load timespan
	t = _T;
	// init sol and add u0
	sol.resize(t.size());
	sol[0] = u0;		
	lin = true;	
	nonlin = true;
}

template<typename T,typename U>
smartPtr<ODE<T,U>> ODE<T,U>::Assemble(MatrixX<T>& _L, funcPtrRef<MatrixX<T>> _N, 
																	 					MatrixX<T>& u0, VectorX<U>& _x, VectorX<U>& _T)
{
	return smartPtr<ODE<T,U>>(new ODE<T,U>(_L,_N,u0,_x,_T));
}
template<typename T,typename U>
smartPtr<ODE<T,U>> ODE<T,U>::Assemble(MatrixX<T>& _L, MatrixX<T>& u0, 
																						VectorX<U>& _x, VectorX<U>& _T)
{
	return smartPtr<ODE<T,U>>(new ODE<T,U>(_L,u0,_x,_T));
}
// construct with 0 nonlinear part
template<typename T,typename U>
ODE<T,U>::ODE(MatrixX<T>& _L, MatrixX<T>& u0, VectorX<U>& _x, VectorX<U>& _T)
{
	if (!std::is_floating_point<T>::value && !(
			std::is_same<T, std::complex<double>>::value ||
			std::is_same<T, std::complex<float>>::value ||
			std::is_same<T,std::complex<long double>>::value))
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " expects floating point type" << std::endl;
		exit(1);
	}
	if (!std::is_floating_point<U>::value && !(
			std::is_same<U, std::complex<double>>::value ||
			std::is_same<U, std::complex<float>>::value ||
			std::is_same<U,std::complex<long double>>::value))
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " expects floating point type" << std::endl;
		exit(1);
	}
	// load L and grid
	L = _L;
	x = _x;
	// set nonlin to 0
	Nfunc = nullptr;
	nonlin = false;
	lin = true;
	// load timespan
	t = _T;
	// init sol and add u0
	sol.resize(t.size());
	sol[0] = u0;			
}
// construct with 0 linear part
template<typename T,typename U>
ODE<T,U>::ODE(funcPtrRef<MatrixX<T>>_N, MatrixX<T>& u0, VectorX<U>& _x, VectorX<U>& _T)
{
	if (!std::is_floating_point<T>::value)
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " expects floating point type" << std::endl;
		exit(1);
	}
	// set lin to 0
	lin = false;
	// load grid
	x = _x;
	// load funcPtr
	Nfunc = _N;
	nonlin = true;
	// load timespan
	t = _T;
	// init sol and add u0
	sol.resize(t.size());
	sol[0] = u0;			
}

// solve vi ODE method
template<typename T, typename U>
void ODE<T,U>::solve(const std::string& solver)
{
	if (solver == "ab2bd2")
	{
		ab2bd2();
	}
	else if (solver == "etdrk2")
	{
		etdrk2();
	}
	else
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " solver type " << solver << " is not supported\n"; 
		exit(1);
	
	}
}

// implementation of ab2bd2
template<typename T,typename U>
void ODE<T,U>::ab2bd2()
{
	T dt = t[1]-t[0];
	if (nonlin & lin)
	{	
		// 2 steps of forward euler to initialize
		sol[1] = sol[0] + dt*(L*sol[0] + Nfunc(sol[0])); 
		MatrixX<T> Nprev = Nfunc(sol[1]);
		sol[2] = sol[1] + dt*(L*sol[1] + Nprev); 
		// define matrix that isolates explicit term
		MatrixX<T> stepMat = MatrixX<T>::Identity(L.rows(),L.cols()) - (2.*dt/3.)*L;
		MatrixX<T> Ncurr;
		for(int i = 3; i < t.size(); ++i)
		{
			Ncurr = Nfunc(sol[i-1]);
			sol[i] = stepMat.colPivHouseholderQr().solve(
								(4./3.)*sol[i-1]-(1./3.)*sol[i-2]+(2.*dt/3.)*(2*Ncurr - Nprev)); 
			Nprev = Ncurr;
		}	
	}
	else if (lin & !nonlin)
	{
		// 2 steps of forward euler to initialize
		std::cout << sol[0].rows() << " " << sol[0].cols() << "\n";

		sol[1] = sol[0] + dt*L*sol[0]; 
		sol[2] = sol[1] + dt*L*sol[1]; 
		// define matrix that isolates explicit term
		MatrixX<T> stepMat = MatrixX<T>::Identity(L.rows(),L.cols()) - (2.*dt/3.)*L;
		for(int i = 3; i < t.size(); ++i)
		{
			sol[i] = stepMat.colPivHouseholderQr().solve(
								(4./3.)*sol[i-1]-(1./3.)*sol[i-2]); 
		}	
	}
	else if (!lin & nonlin)
	{
		// 2 steps of forward euler to initialize
		sol[1] = sol[0] +  dt*Nfunc(sol[0]); 
		MatrixX<T> Nprev = Nfunc(sol[1]);
		sol[2] = sol[1] + dt*Nprev; 
		MatrixX<T> Ncurr; 
		for(int i = 3; i < t.size(); ++i)
		{
			Ncurr = Nfunc(sol[i-1]);
			sol[i] = (4./3.)*sol[i-1]-(1./3.)*sol[i-2]+(2.*dt/3.)*(2*Ncurr - Nprev); 
			Nprev = Ncurr;
		}	
	}
	else
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " linear and nonlinear terms are empty\n"; 
		exit(1);
	}		
}

// implementation of etdrk2
template<typename T,typename U>
void ODE<T,U>::etdrk2()
{
	T dt = t[1]-t[0];
	if (nonlin & lin)
	{
		// precomputing terms for  u = e^(Ldt)u_n + L^(-1)(e^(Ldt)-I)*N(u_n)
		MatrixX<T> pred1_tmp = dt*L;
		// e^(Ldt)
		MatrixX<T> pred1 = pred1_tmp.array().exp();
		// taylor series for expm1 (not available for complex types in eigen)
		MatrixX<T> pred1_tmp_p2, pred1_tmp_p3, pred1_tmp_p4, pred1_tmp_p5;
		pred1_tmp_p2 = pred1_tmp.array().pow(2.)/2.;
		pred1_tmp_p3 = pred1_tmp.array().pow(3.)/6.;
		pred1_tmp_p4 = pred1_tmp.array().pow(4.)/24.;
		pred1_tmp_p5 = pred1_tmp.array().pow(5.)/120.;
		// e^(Ldt)-I
		MatrixX<T> pred2_tmp = pred1_tmp + pred1_tmp_p2 + pred1_tmp_p3 + pred1_tmp_p4 + pred1_tmp_p5;
		// L^(-1)(e^(Ldt)-I))
		MatrixX<T> pred2 = L.colPivHouseholderQr().solve(pred2_tmp);
		MatrixX<T> corr_tmp = L.array().pow(2.); pred2_tmp = (pred2_tmp)*(1./dt)-L;
		// L^(-2)(e^(Ldt)-I-Ldt)/dt
		MatrixX<T> corr = corr_tmp.colPivHouseholderQr().solve((pred2_tmp));
		MatrixX<T> predictor, corrector,Nsoli;
		for (int i = 1; i < t.size(); ++i)
		{
			Nsoli = Nfunc(sol[i-1]);
			predictor = pred1*sol[i-1] + pred2*Nsoli;
			sol[i] = predictor + corr*(Nfunc(predictor)-Nsoli);
			std::cout << sol[i] << std::endl;
		}
	}
	else if (lin & !nonlin)
	{
		// u = e^(Ldt)u_n + L^(-1)(e^(Adt)-I)*N(u_n)
		MatrixX<T> pred1 = (dt*L).array().exp();
		for (int i = 1; i < t.size(); ++i)
		{
			sol[i] = pred1*sol[i-1];
		}
	}
	else if (!lin & nonlin)
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " scheme doesn't work without linear term\n";
		exit(1);
	}
	else
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " linear and nonlinear terms are empty\n"; 
		exit(1);
	}		
}
