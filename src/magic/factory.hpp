#pragma once

#include <unordered_map>
#include <string>


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

#if defined(HAS_TEMPLATE_TEMPLATE)
    #define REGISTER_FACTORY_(BaseType, DataType) \
        template <template <typename> class SubType, typename... Params> \
        struct get ## BaseType ## _t<DataType, SubType, Params...> { \
            static inline BaseType<DataType>* _(Params... params) \
            { \
                return Factory<BaseType<DataType>>::Register::get<SubType<DataType>>(); \
            } \
        };

    #define REGISTER_FACTORY(BaseType) \
        template <typename DType, template <typename> class SubType, typename... Params> \
        struct get ## BaseType ## _t { \
            static inline BaseType<DType>* _(Params... params); \
        }; \
        REGISTER_FACTORY_(BaseType, float) \
        REGISTER_FACTORY_(BaseType, double); \
        template <typename DType, template <typename> class SubType, typename... Params> \
        BaseType<DType>* get ## BaseType(Params... params) { \
            return get ## BaseType ## _t<DType, SubType>::_(params...); \
        }
    
    #define FromFactory(BaseType, SubType, DType) get ## BaseType<DType, SubType>()
    
#else
    #define REGISTER_FACTORY_(BaseType, DataType) \
        template <class SubType, typename... Params> \
        struct get ## BaseType ## _t<DataType, SubType, Params...> { \
            static inline BaseType<DataType>* _(Params... params) \
            { \
                return Factory<BaseType<DataType>>::Register::get<SubType>(); \
            } \
        };

    #define REGISTER_FACTORY(BaseType) \
        template <typename DType, class SubType, typename... Params> \
        struct get ## BaseType ## _t { \
            static inline BaseType<DType>* _(Params... params); \
        }; \
        REGISTER_FACTORY_(BaseType, float) \
        REGISTER_FACTORY_(BaseType, double); \
        template <typename DType, class SubType, typename... Params> \
        BaseType<DType>* get ## BaseType(Params... params) { \
            return get ## BaseType ## _t<DType, SubType>::_(params...); \
        }
    
    #define FromFactory(BaseType, SubType, DType) get ## BaseType<DType, SubType<DType>>()
#endif
