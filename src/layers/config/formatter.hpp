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

#define NAMESPACED_FORMATTER(Name, NType, DType) \
	namespace NType { namespace Config { \
		inline ::ExtConfig::Name Name(DType t) { return ::ExtConfig::Name(t); } \
	} }

#define TEMPLATED_FORMATTER(Name, DType) \
	namespace ExtConfig { \
		class Name : public Formatter<DType> { \
		public: \
			Name() : Formatter<DType>() {} \
			Name(DType value) : Formatter<DType>(value) {} \
		}; \
	} \
	NAMESPACED_FORMATTER(Name, Float, DType) \
	NAMESPACED_FORMATTER(Name, Double, DType) 


#define NAMESPACED_FORMATTER_EXT(Name, NType, DType) \
	namespace NType { namespace Config { \
		template <typename... T> \
		inline ::ExtConfig::Name<DType> Name(T... t) { return ::ExtConfig::Name<DType>(t...); } \
	} }
	

#define TEMPLATED_FORMATTER_EXT(Name, ExtDType) \
	namespace ExtConfig { \
		template <typename DType> \
		class Name : public Formatter<ExtDType> { \
		public: \
			Name() : Formatter<ExtDType>() {} \
			Name(ExtDType value) : Formatter<ExtDType>(value) {} \
		}; \
	} \
	NAMESPACED_FORMATTER_EXT(Name, Float, float) \
	NAMESPACED_FORMATTER_EXT(Name, Double, double) 

#define DEFAULT_FORMATTER(Name) \
	TEMPLATED_FORMATTER_EXT(Name, DType);

#define EXTENDED_FORMATTER(Name, Ext) \
	TEMPLATED_FORMATTER_EXT(Name, Ext<DType>);

#define EXTENDED_FORMATTER_PTR(Name, Ext) \
	TEMPLATED_FORMATTER_EXT(Name, Ext<DType>*);
