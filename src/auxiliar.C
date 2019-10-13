#include<auxiliar.H>
#include<algorithm>
#include<iostream>
// convert c++ vector to Eigen 
template<typename T>
VectorX<T> toEigen(std::vector<T>& x)
{
	return Eigen::Map<VectorX<T>,Eigen::Unaligned>(x.data(),x.size());
}
template<typename T>
VectorX<T> toEigen(std::vector<T>&& x)
{
	return Eigen::Map<VectorX<T>,Eigen::Unaligned>(x.data(),x.size());
}
template<typename T>
std::vector<T> toCVec(VectorX<T>& x)
{
	return std::vector<T>(x.data(),x.data()+x.size());
}
template<typename T>
std::vector<T> toCVec(VectorX<T>&& x)
{
	return std::vector<T>(x.data(),x.data()+x.size());
}

// returns func(t0:dt:tf)
template<typename T>
VectorX<T> range(T t0, T tf, T dt, funcPtr<T> func)
{
	// number of points
	int n = (int) std::ceil((T) (tf-t0)/dt)+1;
	VectorX<T> samples = VectorX<T>::LinSpaced(n,t0,t0+dt*(n-1));
	if (func == nullptr) return samples;
	return samples.unaryExpr(func);
}

//template<typename T>
//VectorX<T> range(T t0, T tf, T dt, funcPtr<T> func)
//{
//	if (t0 == tf || dt == 0)
//	{
//		std::cerr << __func__ << " nothing to return" << std::endl;
//		exit(1);
//	}
//	VectorX<T> samples;
//
//	int i = 0;
//	while (t0 <= tf && dt > 0 || t0 >= tf && dt < 0 )
//	{
//		samples.conservativeResize(i+1);
//		samples(i) = func(t0);
//		t0 = t0 + dt;
//		i++;
//	}
//	std::cout << "samples: " << samples.size() << std::endl;
//	return samples;
//}


