#ifndef FFT_H // compiler guards
#define FFT_H

#include<iostream>
#include<type_traits>
#include<unsupported/Eigen/FFT>
#include<auxiliar.H> // Eigen, func_ptr, range

/*
	This object is built on Eigen's wrapper of FFTW. Instantiating it with
	real floating point type computes the forward transform of the data. 
	Instantiating it with complex floating point types computes the inverse
  transform of the data. That is, it treats the data as if it was already
	forward transformed. This ambiguity can be amended by templating by an 
	additional type (todo).
*/

template<typename T>
class FFT
{
	public:
		// default ctor
		FFT():shifted(false){};
		// construct with data and compute forward transform
		FFT(VectorX<T>& _data);
		// construct with data and forward transform
		FFT(VectorX<T>& _data, VectorX<std::complex<T>>& _freq);
		// construct from func_ptr, t0, tf and number of points n
		FFT(funcPtr<T> func, T t0, T tf, int n);
		// construct with forward transform and compute data
		FFT(VectorX<std::complex<T>>& _freq);
		FFT(VectorX<std::complex<T>>&& _freq);
		// initialize and type check
		void init();
		
		// set attribute routines (resets to default to state and then sets)
		// set data and compute fwd transform to get freq
		void setData(const VectorX<T>& _data);
		// set freq and compute inv transform to get data
		void setFreq(const VectorX<std::complex<T>>& _freq);
		
		// get attribute routines 
		VectorX<T> getData();
		VectorX<std::complex<T>> getFreq(); 
		VectorX<T> getInterpData();
		
		// fftshift for plotting purposes, swaps halves of freq. no-op if already shifted
		void fftshift();	
		static void fftshift(VectorX<T>& x);	
		// ifftshift swaps swapped halves of freq back to original no-op if not shifted
		void ifftshift();
		static void ifftshift(VectorX<T>& x);	
		
		// spectral interpolation onto grid of size m>data.size() via zero padding
		//load data and interpolate
		void interpolate(const VectorX<T>& _data, int m); 	
		// interpolate on already loaded data
		void interpolate(int m);	


		// empty dtor since memory is managed already
		~FFT(){};
		
		// reset attributes to default state
		void reset();

	private:
		VectorX<T> data;
		VectorX<std::complex<T>> freq;
		VectorX<T> interpData;
		Eigen::FFT<T> fft;
		bool shifted;
};

#include "../src/FFT.C"

#endif
