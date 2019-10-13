#include<FFT.H> // Eigen headers, FFT module wrapping Eigen::FFT, std::vector and custom type decls
#include<Eigen/StdVector> // aligned_allocator to contain eigen objects in std::vector
#include<memory> // exposes smart_ptr classes 
#include<matplotlibcpp.h> // wrapper around python matplotlib
#include<ODE.H>

// smart pointer for FFT<double> class
typedef std::shared_ptr<FFT<double>> fftPtr;
typedef ODE<std::complex<double>,double> ode;
typedef smartPtr<ode> odePtr;

// like import as plt
namespace plt = matplotlibcpp;

double phi0_1(double x)
{
	//return (double) (10.+x)*(x < 0 && x >= -10);
	return 0.5*std::pow(1./std::cosh(x*0.5),2);
}

VectorX<double> phi0_1true(double x0, double xf, double dx, double t)
{
	return range<double>(x0-t,xf-dx-t,dx, &phi0_1);
}

// non-linear term
MatrixX<std::complex<double>>  nonlin(MatrixX<std::complex<double>>& phi_hat)
{
	double L = 60.;
	const std::complex<double> i(0.0,1.0);
	int m = phi_hat.rows();
	int n = phi_hat.cols();
	
	VectorX<double> modes = 2.*M_PI*range<double>(-m/2.,m/2.-1.,1.)/L; 

	FFT<double>::ifftshift(modes);
	// set n/2 mode to 0 for even n 
	if (m % 2 == 0)
	{
		modes(m/2) = 0;
	}

	MatrixX<std::complex<double>> k = modes.asDiagonal();
		
	VectorX<double> phi2;
	VectorX<std::complex<double>> phi_hatj;
	MatrixX<std::complex<double>> B(m,n);
	for (int j = 0; j < n; ++j)
	{
		phi_hatj = phi_hat.col(j);	
		fftPtr invfft = fftPtr(new FFT<double>(phi_hatj));
		phi2 = invfft->getData().array().pow(2.);	
		fftPtr fwdfft = fftPtr(new FFT<double>(phi2));
		B.col(j) = -3.*i*k*fwdfft->getFreq();
	}
	return B;			
}

MatrixX<double> nonlin2(MatrixX<double>& u)
{
	MatrixX<double> u2 = u.array().pow(2);
	return u2;
}

//int main()
//{
//	typedef ODE<double,double> ode;
//	typedef smartPtr<ode> odePtr;
//	// u_t = au + nu
//	MatrixX<double> A(1,1); A(0,0) = -1.;
//	MatrixX<double> u0(1,1); u0(0,0) = 0.5; 
//	VectorX<double> x(1);
//	VectorX<double> t(VectorX<double>::LinSpaced(10000,0,1));
//	odePtr ode = ode::Assemble(A,&nonlin2,u0,x,t);
//	ode->solve("ode");
//	TensorX3<double> sol = ode->getSol();
//	std::vector<double> vsol(sol.size());
//	for (int i = 0; i < sol.size(); ++i)
//	{
//		vsol[i] = sol[i](0);
//	}	
//	plt::named_plot("approx",toCVec(t),vsol,"bo");
//	VectorX<double> 
//		trueSol(t.unaryExpr([u0](double j){return 1./((1./u0(0,0) - 1.)*std::exp(j)+1.);}));
//	plt::named_plot("true",toCVec(t),toCVec(trueSol),"rx");
//	plt::legend();
//	plt::show();
//}

int main()
{
	const std::complex<double> i(0.0,1.0);
	// number of points	
	int n = 256;
	int nt = 1000;
	// first and last time, dx
	double L = 60.;
	double x0,xf,dx; x0=-1*L/2.; xf=L/2.; dx= (double) L/n;
	double t0,tf,dt; t0=0.0; tf=L/4.; 
	
	fftPtr fft = fftPtr(new FFT<double>(&phi0_1,x0,xf,n));
	
	// define initial cond 
	MatrixX<std::complex<double>> phi0_hat_with0 = fft->getFreq();
	// remove 0 mode
	std::complex<double> zero_mode = phi0_hat_with0(0);
	MatrixX<std::complex<double>> phi0_hat = phi0_hat_with0.middleRows(1,n-1);
	
	// get modes
	VectorX<double> modes_with0 = 2*M_PI*range<double>(-n/2.,n/2.-1.,1.)/L; 
	FFT<double>::ifftshift(modes_with0);
	// remove 0 mode
	VectorX<double> modes = modes_with0.tail(n-1);

	// define timespan 
	VectorX<double> t(VectorX<double>::LinSpaced(nt,t0,tf));

	// define Lin term
	VectorX<std::complex<double>> k = i*modes.array().pow(3.);
	MatrixX<std::complex<double>> A = k.asDiagonal();

	// assemble system with nonlin term
	odePtr ode = ode::Assemble(A,&nonlin,phi0_hat,modes,t); 
	ode->solve("etdrk2");
	TensorX3<std::complex<double>> sol = ode->getSol();
	std::vector<double> x(toCVec(range<double>(x0,xf-dx,dx)));

	for (int i = 0; i < sol.size(); ++i)
	{
		plt::clf();
		VectorX<std::complex<double>> ftsolAtTj = sol[i].col(0); 
		VectorX<std::complex<double>> ftSolAtTj_with0(n);
		ftSolAtTj_with0.tail(n-1) = ftsolAtTj; ftSolAtTj_with0(0) = zero_mode; 
		fftPtr ifft = fftPtr(new FFT<double>(ftSolAtTj_with0));
		std::vector<double> phiTrue = toCVec(phi0_1true(x0,xf,dx,t(i)));
//		plt::named_plot("true",x, phiTrue);
		plt::named_plot("approx", toCVec(range<double>(x0,xf-dx,dx)),toCVec(ifft->getData()));
		plt::legend();
		plt::pause(1);
	}
}

