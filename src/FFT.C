#include<FFT.H>
#include<complex>
// construct with data and compute forward transform
template<typename T>
FFT<T>::FFT(VectorX<T>& _data)
{
	// checks
	init();
	// copy ref to data
	data = _data;
	// get fft and save to freq
	fft.fwd(freq,data);	
}
// construct with forward transform and compute data with inverse
template<typename T>
FFT<T>::FFT(VectorX<std::complex<T>>& _freq)
{
	init();
	freq = _freq;
	fft.inv(data,freq);
}
template<typename T>
FFT<T>::FFT(VectorX<std::complex<T>>&& _freq)
{
	init();
	freq = _freq;
	fft.inv(data,freq);
}

// construct with data and forward transform
template<typename T>
FFT<T>::FFT(VectorX<T>& _data, VectorX<std::complex<T>>& _freq)
{
	// checks
	init();
	if (_data.size() != _freq.size())
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " size mismatch in data and forward transform\n"; 
		exit(1);
	}
	data = _data;
	freq = _freq;
}
// construct from func_ptr, t0, tf and number of points n
template<typename T>
FFT<T>::FFT(funcPtr<T> func, T t0, T tf, int n)
{
	// checks
	init();
	// create n point linear sampling of func for x in [t0,tf)
	T dt = (T) (tf-t0)/n;
	data = range<T>(t0,tf-dt,dt,func);
	// get forward transform
	fft.fwd(freq,data);
}

// initialize and type check
template<typename T>
void FFT<T>::init()
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
	shifted = false;
}

// reset attributes to default state
template<typename T>
void FFT<T>::reset()
{
	data.resize(0);
	freq.resize(0);
	interpData.resize(0);
	shifted = false;
}

// set attribute routines (resets to default to state and then sets)
template<typename T>
void FFT<T>::setData(const VectorX<T>& _data)
{
	reset();
	data = _data;
	fft.fwd(freq,data);
}
template<typename T>
void FFT<T>::setFreq(const VectorX<std::complex<T>>& _freq)
{
	reset();
	freq = _freq;
	fft.inv(data,freq);
}
// get attribute routines 
template<typename T>
VectorX<T> FFT<T>::getData()
{
	if (data.size() == 0)
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " Warning: data is empty" << std::endl;
	}
	return data;
}
template<typename T>
VectorX<std::complex<T>> FFT<T>::getFreq()
{
	if (freq.size() == 0)
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " Warning: freq is empty" << std::endl;
	}
	return freq;
} 

template<typename T>
VectorX<T> FFT<T>::getInterpData()
{
	if (interpData.size()==0)
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " Warning: interpData is empty" << std::endl;
	}
	return interpData;
} 

// interpolate on already loaded data
template<typename T>
void FFT<T>::interpolate(int m)
{
	if (interpData.size()>0)
	{
		interpData.resize(0);
	}
	if(data.size()==0)
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " data is empty" << std::endl;
		std::cerr << "try calling with interpolate(data,m)" << std::endl;
		exit(1);
	}
	int n = data.size();
	if (n > m)
	{
		std::cerr << __LINE__ << " " << __FILE__ << std::endl;
		std::cerr << __func__ << " n must be <= m" << std::endl;
		exit(1);
	}
	if (freq.size() == 0)
	{
		// compute forward transform
		fft.fwd(freq,data);	
	}
	if (shifted) ifftshift();
	// nyquist bin
	int mid = std::ceil((n+1.)/2.);
	// pad zeros in middle to increase sampling in time domain
	VectorX<std::complex<T>> freq_pad = VectorX<std::complex<T>>::Zero(m);
	freq_pad.head(mid) = freq.head(mid);
	freq_pad.tail(n-mid) = freq.tail(n-mid);
	// deal with unmatched middle mode for even n
	if (n%2==0)
	{
		freq_pad(mid-1) = freq_pad(mid-1)/((T) 2.0);
		freq_pad(mid+m-n-1) = freq_pad(mid-1);
	}
	// invert to get interpolant
	fft.inv(interpData,freq_pad);
	// adjust for gain loss by multiplying by m/n
	interpData = interpData*m/n;
}

// spectral interpolation onto grid of size m>data.size() via zero padding
//	load data and interpolate 
template<typename T>
void FFT<T>::interpolate(const VectorX<T>& _data, int m)	
{
	reset();
	data = _data;
	interpolate(m);
}

// fftshift for plotting purposes, swaps halves of freq
template<typename T>
void FFT<T>::fftshift(VectorX<T>& x)
{
	// number of modes
	int n = x.size();
	// midpoint index of array
	int mid = (int) std::floor(n/2.);
	if (n % 2 == 0)
	{
		x.head(mid).swap(x.tail(mid));
	}
	else
	{
		VectorX<T> tmp = x.head(mid);
		x.head(n-mid) = x.tail(n-mid);
		x.tail(mid) = tmp;
	}
}

template<typename T>
void FFT<T>::fftshift()
{
	if (!shifted)
	{
		FFT<std::complex<T>>::fftshift(freq);
		shifted = true;
	}
}


template<typename T>
void FFT<T>::ifftshift(VectorX<T>& x)
{
	// number of modes
	int n = x.size();
	// midpoint index of array
	int mid = (int) std::floor(n/2.);
	if (n % 2 == 0)
	{
		x.head(mid).swap(x.tail(mid));
	}
	else
	{
		VectorX<T> tmp = x.head(mid+1);
		x.head(n-mid-1) = x.tail(n-mid-1);
		x.tail(mid+1) = tmp;
	}
}

// ifftshift swaps swapped halves of freq back to original
template<typename T>
void FFT<T>::ifftshift()
{
	if(shifted)
	{
		FFT<std::complex<T>>::ifftshift(freq);	
		shifted = false;
	}
}

// explicit instantiation of allowed types by which the class can be templated
//template class FFT<float>;
//template class FFT<double>;
//template class FFT<long double>;
