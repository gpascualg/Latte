#pragma once

#include <unordered_map>
#include <string>


template <typename DType>
class Matrix;


template <class BaseType>
class Factory
{
public:
    class Register
    {
    public:
        template <class SubType, typename... Params>
        static BaseType* get(Params... params)
        {
            auto it = _instances.find(SubType::FactoryName());
            BaseType* instance = nullptr;
            
            if (it == _instances.end())
            {
                instance = new SubType(params...);
                _instances.insert(std::make_pair(SubType::FactoryName(), instance));
            }
            else
            {
                instance = it->second;
            }
            
            return instance;
        }
    };
    
private:
    static std::unordered_map<std::string, BaseType*> _instances;
};

template <class BaseType>
std::unordered_map<std::string, BaseType*> Factory<BaseType>::_instances;


template <typename DType>
class Activation
{
public:
	inline void apply(Matrix<DType>* matrix)
	{
		apply(matrix, matrix);
	}

	inline void derivative(Matrix<DType>* matrix, Matrix<DType>* alpha)
	{
		derivative(matrix, matrix, alpha);
	}

	virtual void apply(Matrix<DType>* matrix, Matrix<DType>* dest) = 0;
	virtual void derivative(Matrix<DType>* matrix, Matrix<DType>* dest, Matrix<DType>* alpha) = 0;
    
protected:
	Activation() {};
};
