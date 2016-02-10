#pragma once

template <typename DType>
class Matrix;

struct Shape;

template <typename DType>
class Activation;

template <typename DType>
class Layer
{
public:
	Layer(Shape shape, int num_output, Activation<DType>* activaton);
	virtual ~Layer();

	virtual Matrix<DType>* forward(Matrix<DType>* in);
	virtual Matrix<DType>* backward(Matrix<DType>* error);
	virtual void update();

	inline Matrix<DType>* W();
	inline Shape inShape();
	inline Shape outShape();

protected:
	Activation<DType>* _activaton;

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
