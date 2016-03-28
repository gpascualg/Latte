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
	struct LayerConnection
	{
		LayerConnection(Matrix<DType>* input):
			input(input)
		{}

		Layer<DType>* layer;
		Matrix<DType>* input;

		Matrix<DType>* error;

		Matrix<DType>** weights;
		Matrix<DType>** bias_weights;
		Matrix<DType>** output;
		Matrix<DType>** delta;
	};

	template <typename DType>
	class FinalizedLayer
	{
		template <typename T>
		friend class Layer;

	protected:
		FinalizedLayer(Layer<DType>* layer);

	public:
		FinalizedLayer(const FinalizedLayer<DType>& layer);

	public:
		inline Layer<DType>* operator->() { return _layer; }

		inline FinalizedLayer<DType>& operator<<(Matrix<DType>& matrix)
		{
			*_layer << matrix;
			return *this;
		}

		inline FinalizedLayer<DType>& operator<<(Layer<DType>& layer)
		{
			*_layer << layer;
			return *this;
		}

		inline FinalizedLayer<DType>& operator<<(FinalizedLayer<DType>& layer)
		{
			*_layer << *layer._layer;
			return *this;
		}

		inline Layer<DType>* asd() { return _layer; }

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

		Layer<DType>& operator<<(Matrix<DType>& other);
		Layer<DType>& operator<<(Layer<DType>& other);
		inline Layer<DType>& operator<<(FinalizedLayer<DType>& other)
		{
			return (*this << *other._layer);
		}

		Layer<DType>& operator<<(Config::Data<DType>&& data)
		{
			_isFirst = true;
			return (*this << *data());
		}

		FinalizedLayer<DType> operator<<(Config::Finalizer&& f)
		{
			LATTE_ASSERT("Layer not ready, all should be 1:" <<
				std::endl << "\tNumOutput: " << _numOutput.isSet() <<
				std::endl << "\tFiller: " << _filler.isSet() << 
				std::endl << "\tActivation: " << _activation.isSet(),
				_numOutput.isSet() && _filler.isSet() && _activation.isSet());

			return FinalizedLayer<DType>(this);
		}

	public:
		bool canBeForwarded();
		bool isLast();

		virtual Matrix<DType>* forward();
		virtual std::vector<Matrix<DType>*> backward(std::vector<Matrix<DType>*> errors);
		virtual void update(float learningRate);

		inline Shape inShape() { return _inShape(); }
		inline Shape outShape() { return _output[0]->shape(); }
		inline std::vector<Matrix<DType>*>& output() { return _output; }
		inline std::vector<Matrix<DType>*>& W() { return _weights; }

	protected:
		Config::NumOutput _numOutput;
		Config::Dropout _dropout;
		Config::Shape _inShape;
		Config::Bias<DType> _bias;
		Config::Filler<DType> _filler;
		Config::Activation<DType> _activation;

    	Matrix<DType>* _bias_values;
		//Matrix<DType>* _diff;

		std::vector<Matrix<DType>*> _weights;
    	std::vector<Matrix<DType>*> _bias_weights;
		std::vector<Matrix<DType>*> _output;
    	std::vector<Matrix<DType>*> _delta;

    	bool _isFirst;
		bool _forwardDone;
		int _maxInputs;
		int _forwardsTo;
		std::vector<LayerConnection<DType>*> _inputs;
	};


	template <typename DType>
	class LayerWrapper
	{
	public:
		LayerWrapper(Layer<DType>* layer):
			_layer(layer)
		{}

		~LayerWrapper()
		{
			if (_layer)
			{
				delete _layer;
			}
		}

		template <typename T>
		LayerWrapper<DType>& operator<<(T &&val)
		{
			*_layer << std::move(val);
			return *this;
		}

		FinalizedLayer<DType> operator<<(Config::Finalizer&& fin)
		{
			auto layer = _layer;
			_layer = nullptr;
			return (*layer << std::move(fin));
		}

	private:
		Layer<DType>* _layer;
	};
}

#define REGISTER_LAYER_I(LayerName, NType, DType) \
	namespace NType { \
		inline ::Layer::LayerWrapper<DType> LayerName() { return ::Layer::LayerWrapper<DType>(new ::Layer::LayerName<DType>()); } \
	}

#define REGISTER_LAYER(LayerName) \
	REGISTER_LAYER_I(LayerName, Float, float) \
	REGISTER_LAYER_I(LayerName, Double, double)

