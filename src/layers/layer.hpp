#pragma once

#include <type_traits>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <queue>

// This should be forward declared and methods declared on a cpp
#include "common.hpp"
#include "layers/config/formatter_specialize.hpp"

template <typename DType>
class Matrix;


namespace Layer
{
	template <typename T>
	class Layer;

	template <typename T>
	class FinalizedLayer;


	template <typename DType>
	class FinalizedLayer
	{
		template <typename T>
		friend class Layer;

	protected:
		FinalizedLayer(Layer<DType>* layer);

	public:
		inline Layer<DType>* operator->() { return _layer; }

	private:
		Layer<DType>* _layer;
	};


	template <typename DType>
	class Layer
	{
		template <typename T>
		friend class FinalizedLayer;

	public:
		Layer();
		virtual ~Layer();

		Layer<DType>& operator<<(Config::Shape&& shape)
		{
			_inShape = shape;
			return *this;
		}

		Layer<DType>& operator<<(Config::Bias<DType>&& bias)
		{
			_bias = bias;
			return *this;
		}

		Layer<DType>& operator<<(Config::NumOutput&& numOutput)
		{
			_numOutput = numOutput;
			return *this;
		}
		
		Layer<DType>& operator<<(Config::Dropout&& dropout)
		{
			_dropout = dropout;
			return *this;
		}
		
		Layer<DType>& operator<<(Config::Filler<DType>&& filler)
		{
			_filler = filler;
			return *this;
		}
		
		Layer<DType>& operator<<(Config::Activation<DType>&& activation)
		{
			_activation = activation;
			return *this;
		}

		FinalizedLayer<DType> operator<<(Config::Finalizer&& f)
		{
			LATTE_ASSERT("Layer not ready, all should be 1:" <<
				std::endl << "\tNumOutput: " << _numOutput.isSet() <<
				std::endl << "\tShape: " << _inShape.isSet() << 
				std::endl << "\tFiller: " << _filler.isSet() << 
				std::endl << "\tActivation: " << _activation.isSet(),
				_numOutput.isSet() && _inShape.isSet() && _filler.isSet() && _activation.isSet());

			return FinalizedLayer<DType>(this);
		}

	public:
		virtual Matrix<DType>* forward();
		virtual Matrix<DType>* backward(Matrix<DType>* error);
		virtual void update(DType learning_rate);

		void connect(FinalizedLayer<DType>& layer);
		void connect(Matrix<DType>* data);

		inline Matrix<DType>* W() { return _weights; }
		inline Shape inShape() { return _inShape(); }
		inline Shape outShape() { return _output->shape(); }
		inline Matrix<DType>* output() { return _output; }

	protected:
		Config::NumOutput _numOutput;
		Config::Dropout _dropout;
		Config::Shape _inShape;
		Config::Bias<DType> _bias;
		Config::Filler<DType> _filler;
		Config::Activation<DType> _activation;

		Matrix<DType>* _in;
		Matrix<DType>* _weights;
    	Matrix<DType>* _bias_weights;
    	Matrix<DType>* _bias_values;
		Matrix<DType>* _output;
		Matrix<DType>* _delta;
		Matrix<DType>* _diff;
	};


	template <typename DType>
	class LayerWrapper
	{
	public:
		LayerWrapper(Layer<DType>* layer):
			_layer(layer)
		{}

		template <typename T>
		LayerWrapper<DType>& operator<<(T val)
		{
			return (*_layer << std::move(val));
		}

		FinalizedLayer<DType> operator<<(Config::Finalizer&& fin)
		{
			_layer = nullptr;
			return (*_layer << fin);
		}

	private:
		Layer<DType>* _layer;
	};
}

/*
template <typename DType>
class Layer
{
public:
	class LayerIterator
	{
		friend class Layer;

	protected:
		LayerIterator(Layer<DType>* layer);

	public:
		void operator++();
		void operator--();
		Layer<DType>* operator*();
		bool next();
		Layer<DType>* last();

	private:
		Layer<DType>* _current;
		Layer<DType>* _previous;
		std::queue<std::pair<Layer<DType>*, Layer<DType>*> > _queue;
	};

public:
	Layer();
	virtual ~Layer();

	virtual Matrix<DType>* forward();
	virtual Matrix<DType>* backward(Matrix<DType>* error);
	virtual void update(DType learning_rate);

	inline Matrix<DType>* W() { return _weights; }
	inline Shape inShape() { return _in_shape; }
	inline Shape outShape() { return _out_shape; }
	inline Matrix<DType>* output() { return _output; }
	
	void connect(Layer<DType>* layer);
	void connect(Matrix<DType>* data);
	LayerIterator iterate();
};
*/
