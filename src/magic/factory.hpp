#pragma once

#include <vector>
#include <string>
#include "latte_compiler_detection.h"


class AutoCleanable
{
public:
    virtual ~AutoCleanable() { clean(); }

protected:
    virtual void clean() {};

protected:
    static std::vector<AutoCleanable*> _singletons;

private:
    static AutoCleanable _instance;
};



template <class BaseType, class SubType, typename... Params>
class Factory : public AutoCleanable
{
public:
    static BaseType* get(Params... params)
    {
        if (_instance == nullptr)
        {
            _singletons.emplace_back(new Factory<BaseType, SubType, Params...>(params...));
        }
        
        return _instance;
    }

protected:
    void clean() override
    {
        delete _instance;
    }

private:
    Factory(Params... params)
    {
        _instance = new SubType(params...);
    }
    
private:
    static BaseType* _instance;
};

template <class BaseType, class SubType, typename... Params>
BaseType* Factory<BaseType, SubType, Params...>::_instance = nullptr;


#define REGISTER_FACTORY_NS(BaseType, NType, DType) \
    namespace NType \
    { \
        template <template <typename T> class SubType, typename... Args> \
        inline BaseType<DType>* Get ## BaseType(Args... args) \
        { \
            return From ## BaseType ## Factory<DType, SubType, Args...>(args...); \
        } \
    }


#define REGISTER_FACTORY(BaseType) \
    template <typename DType, template <typename DType> class T, typename... Args> \
    inline BaseType<DType>* From ## BaseType ## Factory(Args... args) \
    { \
        return FromFactory<BaseType<DType>, T<DType>, Args...>(args...); \
    } \
    REGISTER_FACTORY_NS(BaseType, Float, float) \
    REGISTER_FACTORY_NS(BaseType, Double, double)


template <class BaseType, class SubType, typename... Params>
inline BaseType* FromFactory(Params... params)
{
    return Factory<BaseType, SubType, Params...>::get(params...);
}
