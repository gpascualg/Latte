#pragma once

#include <utility>
#include <vector>
#include <queue>

// This should be forward declared and methods declared on a cpp
#include "common.hpp"

template <typename DType>
class Matrix;

struct Shape;

template <typename DType>
class Activation;

template <typename DType>
class Filler;

template <typename DType>
class Layer;


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
		inline void operator++();
		inline void operator--();
		inline Layer<DType>* operator*();
		inline bool next();
		inline Layer<DType>* last();

	private:
		Layer<DType>* _current;
		Layer<DType>* _previous;
		std::queue<std::pair<Layer<DType>*, Layer<DType>*> > _queue;
	};

public:
	Layer(Shape shape, int num_output, Activation<DType>* activaton, Filler<DType>* filler);
	virtual ~Layer();

	virtual Matrix<DType>* forward();
	virtual Matrix<DType>* backward(Matrix<DType>* error);
	virtual void update();

	inline Matrix<DType>* W();
	inline Shape inShape();
	inline Shape outShape();
	inline Matrix<DType>* output() { return _output; }
	
	inline void connect(Layer<DType>* layer) { layer->_next.push_back(this); _previous.push_back(layer); _in = layer->_output; }
	inline void connect(Matrix<DType>* data) { _in = data; }
	inline LayerIterator iterate() { return LayerIterator(this); }

protected:
	Activation<DType>* _activaton;

	std::vector<Layer<DType>*> _next;
	std::vector<Layer<DType>*> _previous;

	Matrix<DType>* _in;
	Matrix<DType>* _weights;
	Matrix<DType>* _output;
	Matrix<DType>* _delta;
	Matrix<DType>* _diff;

	Shape _in_shape;
	Shape _out_shape;
};

template <typename DType> Matrix<DType>* Layer<DType>::W();
template <typename DType> Shape Layer<DType>::inShape();
template <typename DType> Shape Layer<DType>::outShape();

/*
template <typename DType> void Layer<DType>::LayerIterator::operator++();
template <typename DType> void Layer<DType>::LayerIterator::operator--();
template <typename DType> Layer<DType>* Layer<DType>::LayerIterator::operator*();
template <typename DType> bool Layer<DType>::LayerIterator::next();
template <typename DType> Layer<DType>* Layer<DType>::LayerIterator::last();
*/
