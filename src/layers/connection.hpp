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
		LayerConnection(Matrix<DType>* input);
		LayerConnection(Layer<DType>* input);
		~LayerConnection();

		Layer<DType>* layer;

		Matrix<DType>* input;
		Matrix<DType>* weights;
		Matrix<DType>* bias_weights;
		Matrix<DType>* error;
		Matrix<DType>* delta;

		Matrix<DType>** output;
	};

	template <typename DType>
	struct BackwardConnection
	{
		BackwardConnection(Matrix<DType>*& error, 
			Matrix<DType>* delta, Matrix<DType>* weights);
		~BackwardConnection();

		Matrix<DType>*& error;
		Matrix<DType>* delta;
		Matrix<DType>* weights;
	};
}
