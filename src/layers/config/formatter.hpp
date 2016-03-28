#pragma once

template <typename DType>
class Formatter_priv
{
protected:
	Formatter_priv() :
		_set(false)
	{}

	Formatter_priv(DType value) :
		_value(value),
		_set(true)
	{}

public:
	inline bool isSet() { return _set; }
	inline DType& operator()() { return _value; }

	Formatter_priv<DType>& operator=(Formatter_priv<DType>& other)
	{
		_set = other._set;
		_value = other._value;
		return *this;
	}

protected:
	DType _value;
	bool _set;
};

template <typename DType, typename Sfinae = void>
class Formatter : public Formatter_priv<DType>
{
protected:
	Formatter() :
		Formatter_priv<DType>()
	{}

	Formatter(DType value):
		Formatter_priv<DType>(value)
	{}

public:
	inline DType* operator->()
	{ 
		return &((*this)());
	}
};

template <typename DType>
class Formatter<DType, typename std::enable_if<std::is_pointer<DType>::value>::type> : public Formatter_priv<DType>
{
protected:
	Formatter() :
		Formatter_priv<DType>()
	{}


	Formatter(DType value):
		Formatter_priv<DType>(value)
	{}

public:
	inline DType operator->()
	{ 
		return (*this)();
	}
};

#define TEMPLATED_FORMATTER(Name, DType) \
	class Name : public Formatter<DType> { \
	public: \
		Name() : Formatter<DType>() {} \
		Name(DType value) : Formatter<DType>(value) {} \
	} //

#define DEFAULT_FORMATTER(Name) \
	template <typename DType> \
	TEMPLATED_FORMATTER(Name, DType)

#define EXTENDED_FORMATTER(Name, Ext) \
	template <typename DType> \
	TEMPLATED_FORMATTER(Name, Ext<DType>)

#define EXTENDED_FORMATTER_PTR(Name, Ext) \
	template <typename DType> \
	TEMPLATED_FORMATTER(Name, Ext<DType>*)
