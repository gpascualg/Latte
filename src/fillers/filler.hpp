#pragma once

#include "magic/factory.hpp"

template <typename DType>
class Matrix;

template <typename DType>
class Filler
{
public:
	virtual ~Filler() {}
	virtual void fill(Matrix<DType>* weights) {}

protected:
	Filler() {};
};


REGISTER_FACTORY(Filler);

/*
template <typename DType, template <typename DType> class T, typename... Args>
inline Filler<float> FromFillerFactory(Args... args)
{
    return FromFactory<Filler, T<DType>, Args...>(args...);
}

namespace Float
{
    template <template <typename DType> class SubType, typename... Args>
    inline Filler<float>* GetFiller(Args... args)
    {
        return FromFillerFactory<float, SubType, Args...>(args...);
    }
}

namespace Double
{
    template <template <typename DType> class SubType, typename... Args>
    inline Filler<double>* GetFiller(Args... args)
    {
        return FromFillerFactory<double, SubType, Args...>(args...);
    }
}
*/
