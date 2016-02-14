#pragma once

#include <type_traits>
#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>
#include <tuple>
#include <utility>
#include <string>


template <typename T>
class Parameter;

class GenericParameter
{
public:
	GenericParameter(bool required = false, bool optional = false) :
		_required(required),
		_optional(optional)
	{}

	template <typename T>
	T as()
	{
		assert(!_required && "Required default parameter not found");
		T value = ((Parameter<T>*)this)->Value;

		// If _required, we should never reach this point
		if (_optional || _required)
		{
			delete this;
		}

		return value;
	}

private:
	bool _required;
	bool _optional;
};

template <typename T>
class Parameter : public GenericParameter
{
public:
	Parameter(std::string name) :
   		GenericParameter(),
		Name(name)
	{}

	static Parameter<T>* make_optional(T value)
	{
		return new Parameter<T>("", value, true);
	}

	Parameter<T> operator=(T value)
	{
		return Parameter<T>(Name, value);
	}

private:
	Parameter(std::string name, T value) :
		GenericParameter(),
        Name(name), Value(value)
	{}

	Parameter(std::string name, T value, bool optional) :
		GenericParameter(false, optional),
        Name(name), Value(value)
	{}

public:
	std::string Name;
	T Value;
};

template<std::size_t I = 0, typename D, typename... Tp>
typename std::enable_if<I == sizeof...(Tp), GenericParameter*>::type
    eval(std::string name, std::tuple<Tp...> tpl, const bool required, D def)
{
    if (required)
    {
        return new GenericParameter(true);
    }

    return Parameter<D>::make_optional(def);
}

template<std::size_t I = 0, typename D, typename... Tp>
typename std::enable_if<I < sizeof...(Tp), GenericParameter*>::type
    eval(std::string name, std::tuple<Tp...> tpl, const bool required, D def)
{
    if (std::get<I>(tpl).Name == name) return &std::get<I>(tpl);
    return eval<I + 1, D, Tp...>(name, tpl, required, def);
}

template<std::size_t I = 0, typename D, typename... Tp>
typename std::enable_if<I < sizeof...(Tp), GenericParameter*>::type
    eval(const char* cname, std::tuple<Tp...> tpl, const bool required, D def)
{
    std::string name(cname);
    
    if (std::get<I>(tpl).Name == name) return &std::get<I>(tpl);
    return eval<I + 1, D, Tp...>(name, tpl, required, def);
}

#define STR_(x) #x
#define TPL(x) std::make_tuple(x...)

#define PARAMETER_(param, tpl, rq, def)		eval(STR_(param), tpl, rq, def)

#define NAMED_REQUIRED(param, args)			PARAMETER_(param, TPL(args), true, nullptr)
#define NAMED_OPTIONAL(param, args, def)	PARAMETER_(param, TPL(args), false, def)

#define ARG_REQUIRED(param)					PARAMETER_(param, TPL(args), true, nullptr)
#define ARG_OPTIONAL(param, def)			PARAMETER_(param, TPL(args), false, def)

struct NamedArguments_t {};
#define NamedArguments NamedArguments_t{}
