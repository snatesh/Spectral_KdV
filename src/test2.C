#define _USE_MATH_DEFINES

#include<FFT.H> // Eigen headers, FFT module wrapping Eigen::FFT, std::vector and custom type decls
#include<Eigen/StdVector> // aligned_allocator to contain eigen objects in std::vector
#include<memory> // exposes smart_ptr classes 
#include<matplotlibcpp.h> // wrapper around python matplotlib

// smart pointer for FFT<double> class
typedef std::shared_ptr<FFT<double>> fftPtr;
// like import as plt
namespace plt = matplotlibcpp;



//tensor3 lin_term(std::shared_ptr<FFT<double>> fft)

double phi0_1(double x)
{
	//return (double) (10.+x)*(x < 0 && x >= -10);
	return 0.5*std::pow(2./(std::exp(x*0.5) + std::exp(-x*0.5)),2);
}

int main()
{
	// testing interpft
	
	// first and last time, dt
	double L = 60.;
	double t0,tf, dt; t0=-1*L/2.; tf=L/2.; 
	int m = 2000;
	double dt2 = (double) L/m;
	VectorX<double> sizes(range<double>(50,1000,5));
	std::vector<double> interpErrors;
	std::vector<double> vsizes;
	int j = 0;
	for (int i = 0; i < sizes.size(); ++i)
	{
		plt::clf();
		int n = sizes(i); dt= (double) L/n;

		fftPtr fft = fftPtr(new FFT<double>(&phi0_1,t0,tf,n));
		
		fft->interpolate(m);

		VectorX<double> interpData(fft->getInterpData());

		//VectorX<double> x(range<double>(t0,tf-dt,dt));		

		VectorX<double> x2(range<double>(t0,tf-dt,dt2));

		interpData.conservativeResize(x2.size());

		VectorX<double> truePhi(x2.unaryExpr(&phi0_1));

		VectorX<double> diff(truePhi - interpData);
		
		double interpError = diff.norm()/truePhi.norm();	
		if (interpError > 1e-3)
		{
			//plt::named_plot("interpSignal",toCVec(x2),toCVec(interpData),"b.");
			//plt::named_plot("trueSignal",toCVec(x2), toCVec(truePhi));
			//plt::xlim(-32,32);
			//plt::legend();
			//plt::pause(0.1);
		}	
		else
		{
			interpErrors.push_back(interpError);
			vsizes.push_back(sizes(i));	
		}
	}
	plt::semilogy(vsizes,interpErrors);
	plt::show();
}

