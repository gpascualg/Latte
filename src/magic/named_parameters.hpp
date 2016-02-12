#pragma once

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
		Name(name),
		GenericParameter()
	{}

	static Parameter<T>* make_optional(T value)
	{
		return new Parameter<T>("", value, true);
	}

	Parameter<T> operator=(T value)
	{
		return Parameter<T>(Name, value);
	}

	template<std::size_t I = 0, typename D, typename... Tp>
	inline typename std::enable_if<I == sizeof...(Tp), GenericParameter*>::type
		eval(std::string name, std::tuple<Tp...>& tpl, const bool required, D default) const
	{
		if (required)
		{
			return new GenericParameter(true);
		}

		return Parameter<D>::make_optional(default);
	}

	template<std::size_t I = 0, typename D, typename... Tp>
	inline typename std::enable_if<I < sizeof...(Tp), GenericParameter*>::type
		eval(std::string name, std::tuple<Tp...>& tpl, const bool required, D default) const
	{
		if (std::get<I>(tpl).Name == name) return &std::get<I>(tpl);
		return eval<I + 1, D, Tp...>(name, tpl, required, default);
	}

private:
	Parameter(std::string name, T value) :
		Name(name), Value(value),
		GenericParameter()
	{}

	Parameter(std::string name, T value, bool optional) :
		Name(name), Value(value),
		GenericParameter(false, optional)
	{}

public:
	std::string Name;
	T Value;
};

#define STR_(x) #x
#define TPL(x) std::make_tuple(x...)

#define PARAMETER_(param, tpl, rq, def)		std::get<0>(tpl).eval(STR_(param), tpl, rq, def)

#define NAMED_REQUIRED(param, args)			PARAMETER_(param, TPL(args), true, nullptr)
#define NAMED_OPTIONAL(param, args, def)	PARAMETER_(param, TPL(args), false, def)

#define ARG_REQUIRED(param)					PARAMETER_(param, TPL(args), true, nullptr)
#define ARG_OPTIONAL(param, def)			PARAMETER_(param, TPL(args), false, def)

//#define ARG_REQUIRED(param)					REQUIRED(param, args)
//#define ARG_OPTIONAL(param, def)			OPTIONAL(param, args, def)
