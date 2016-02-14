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
	Layer(Shape shape, int num_output, Activation<DType>* activaton, Filler<DType>* filler);
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

