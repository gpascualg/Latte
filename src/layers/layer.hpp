#pragma once

#include <type_traits>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <queue>

// This should be forward declared and methods declared on a cpp
#include "common.hpp"
#include "layers/connection.hpp"
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

		template <typename T>
		friend class Loss;

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

		Layer<DType>& operator<<(ExtConfig::Shape&& shape)
		{
			_inShape = shape;
			return *this;
		}

		Layer<DType>& operator<<(ExtConfig::Bias<DType>&& bias)
		{
			_bias = bias;
			return *this;
		}

		Layer<DType>& operator<<(ExtConfig::NumOutput&& numOutput)
		{
			_numOutput = numOutput;
			return *this;
		}
		
		Layer<DType>& operator<<(ExtConfig::Dropout&& dropout)
		{
			_dropout = dropout;
			return *this;
		}
		
		Layer<DType>& operator<<(ExtConfig::Filler<DType>&& filler)
		{
			_filler = filler;
			return *this;
		}
		
		Layer<DType>& operator<<(ExtConfig::Activation<DType>&& activation)
		{
			_activation = activation;
			return *this;
		}

		virtual Layer<DType>& operator<<(Layer<DType>& other);

		inline Layer<DType>& operator<<(FinalizedLayer<DType>& other)
		{
			return (*this << *other._layer);
		}

		Layer<DType>& operator<<(ExtConfig::Data<DType>&& data)
		{
			_isFirst = true;
			return (*this << *data());
		}

		FinalizedLayer<DType> operator<<(ExtConfig::Finalizer&& f)
		{
			LATTE_ASSERT("Layer not ready, all should be 1:" <<
				std::endl << "\tNumOutput: " << _numOutput.isSet() <<
				std::endl << "\tFiller: " << _filler.isSet() << 
				std::endl << "\tActivation: " << _activation.isSet(),
				_numOutput.isSet() && _filler.isSet() && _activation.isSet());

			return FinalizedLayer<DType>(this);
		}

	protected:
		Layer<DType>& operator<<(Matrix<DType>& other);

	public:
		bool canBeForwarded();
		virtual bool isLast();

		virtual Matrix<DType>* forward();
		virtual std::vector<BackwardConnection<DType>*> backward();
		virtual void update(float learningRate);

		inline Shape inShape() { return _inShape(); }
		inline Shape outShape() { return _output[0]->shape(); }
		inline std::vector<Matrix<DType>*>& output() { return _output; }
		inline std::vector<LayerConnection<DType>*>& connections() { return _connections; }

	protected:
		ExtConfig::NumOutput _numOutput;
		ExtConfig::Dropout _dropout;
		ExtConfig::Shape _inShape;
		ExtConfig::Bias<DType> _bias;
		ExtConfig::Filler<DType> _filler;
		ExtConfig::Activation<DType> _activation;

    	Matrix<DType>* _bias_values;
		//Matrix<DType>* _diff;

		std::vector<Matrix<DType>*> _output;

    	bool _isFirst;
		bool _forwardDone;
		int _maxInputs;
		int _forwardsTo;
		std::vector<LayerConnection<DType>*> _connections;
	};


	template <typename DType, template <typename DType> class BaseClass>
	class LayerWrapper
	{
	public:
		LayerWrapper(BaseClass<DType>* layer):
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
		LayerWrapper<DType, BaseClass>& operator<<(T &&val)
		{
			*_layer << std::move(val);
			return *this;
		}

		FinalizedLayer<DType> operator<<(ExtConfig::Finalizer&& fin)
		{
			auto layer = _layer;
			_layer = nullptr;
			return (*layer << std::move(fin));
		}

	private:
		BaseClass<DType>* _layer;
	};
}

#define REGISTER_LAYER_I(LayerName, NType, DType, BaseClass) \
	namespace NType { \
		inline ::Layer::LayerWrapper<DType, BaseClass> LayerName() { return ::Layer::LayerWrapper<DType, BaseClass>(new ::Layer::LayerName<DType>()); } \
	}

#define REGISTER_LAYER(LayerName) \
	REGISTER_LAYER_I(LayerName, Float, float, ::Layer::Layer) \
	REGISTER_LAYER_I(LayerName, Double, double, ::Layer::Layer)

#define REGISTER_LOSS(LayerName) \
	REGISTER_LAYER_I(LayerName, Float, float, ::Layer::Loss) \
	REGISTER_LAYER_I(LayerName, Double, double, ::Layer::Loss)
