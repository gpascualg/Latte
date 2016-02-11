#pragma once

template <typename DType>
class Matrix;

template <typename DType>
class Filler
{
public:
	virtual void fill(Matrix<DType>* weights) {};

	template <class AType>
	static Filler<DType>* get()
	{
		if (!_instance)
		{
			_instance = new AType();
		}

		return _instance;
	}

protected:
	Filler() {};

private:
	static Filler<DType>* _instance;
};

template <typename DType>
Filler<DType>* Filler<DType>::_instance = nullptr;

