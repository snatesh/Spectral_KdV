#include<iostream>
#include<iomanip>
#include<type_traits>
#include<algorithm>
#include<vector>
#include<Eigen/Dense>
#include<cmath>
#include<unsupported/Eigen/FFT>
#include<matplotlibcpp.h>

// get and define types and namespaces
using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace plt = matplotlibcpp;

// create templated function ptr type
template<typename T>
using funcPtr = T (*) (T);

// get n linearly spaced samples of func between t0 and tf
template<typename T>
std::vector<T> linSample(funcPtr<T> func, T t0, T tf, int n)
{
	if (!std::is_arithmetic<T>::value)
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " expects arithmetic type" << std::endl;
		exit(1);
	}
	T dt = (tf-t0)/(n-1);
	std::vector<T> samples(n);
	std::generate(samples.begin(),samples.end(),[dt,i=-1,func] () mutable 
	{ 
		++i; 
		return func(dt*i);
	});
	return samples;
}

// fftshift
template<typename T>
void fftshift(std::vector<std::complex<T>>& freq)
{
	if (!std::is_arithmetic<T>::value)
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " expects arithmetic type" << std::endl;
		exit(1);
	}
	// number of modes
	int n = freq.size();
	// midpoint of array
	int mid = (int) std::floor((T) n/2);
	// storage for swap routines
	std::complex<T> tmp;
	// if n even
	if (n%2 == 0)
	{
		for (int i = 0; i < mid; ++i)
		{
			tmp = freq[i];
			freq[i] = freq[mid+i];
			freq[mid+i] = tmp;
		}
	}
	else // if n odd
	{
		tmp = freq[0];
		for (int i = 0; i < mid; ++i)
		{
				freq[i] = freq[mid+i+1];
				freq[mid+i+1] = freq[i+1];
		}
		freq[mid] = tmp;
	}
}

// ifftshift
template<typename T>
void ifftshift(std::vector<std::complex<T>>& freq)
{
	if (!std::is_arithmetic<T>::value)
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " expects arithmetic type" << std::endl;
		exit(1);
	}
	// number of modes
	int n = freq.size();
	// midpoint of array
	int mid = (int) std::floor((T) n/2);
	// storage for swap routines
	std::complex<T> tmp;
	// if n even
	if (n%2 == 0)
	{
		for (int i = 0; i < mid; ++i)
		{
			tmp = freq[i];
			freq[i] = freq[mid+i];
			freq[mid+i] = tmp;
		}
	}
	else // if n odd
	{
		tmp = freq[n-1];
		for (int i = mid-1; i >= 0; i--)
		{
				freq[mid+i+1] = freq[i];
				freq[i] = freq[mid+i];

		}
		freq[mid] = tmp;
	}
}

template<typename T>
std::vector<T> interpft(const std::vector<T>& data, int m)
{
	if (!std::is_arithmetic<T>::value)
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " expects arithmetic type" << std::endl;
		exit(1);
	}
	
	int n = data.size();
	int mid = std::ceil((n+1)/2);
	std::vector<std::complex<T>> freq;
	// instantiate eigen fft
	Eigen::FFT<T> fft;
	// compute forward transform
	fft.fwd(freq,data);	
	// pad zeros in middle to increase sampling in time domain
	freq.insert(freq.begin()+mid,m-n,0.0);
	// deal with unmatched middle mode for even n
	if (n%2==0)
	{
		freq[mid] = freq[mid]/((std::complex<T>) 2);
		freq[mid+m-n] = freq[mid];
	}
	// invert to get interpolant
	std::vector<T> interpData(m);
	fft.inv(interpData,freq);
	// adjust for gain loss by multiplying by m/n
	for (int i = 0; i < m; ++i)
	{
		interpData[i] = interpData[i]*((T)m/n);
	}
	return interpData;
}

int main()
{
	// number of points	
	int n = 100;
	// first and last time, dt
	double t0,tf, dt; t0=0.0; tf=2.*M_PI; dt=(tf-t0)/(n-1);
	// instantiate eigen fft
	Eigen::FFT<double> fft;
	// construct data 
	std::vector<double> data = linSample<double>(&std::sin,t0,tf,n);
	// plot data
	plt::plot(linSample<double>([](double x) {return x;},t0,tf,n),data);
	// declare frequency vector
	std::vector<std::complex<double>> freq;
	fft.fwd(freq,data);
	int m = 200;
	std::vector<double> newdata = interpft<double>(data,m);
	plt::plot(linSample<double>([](double x) {return x;},t0,tf,m),newdata);
	plt::plot(linSample<double>([](double x) {return x;},t0,tf,m),linSample<double>(&std::sin,t0,tf,m));
	plt::show();	
}

