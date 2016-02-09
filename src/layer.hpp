#pragma once

template <typename DType>
class Matrix;

struct Shape;

template <typename DType>
class Layer
{
public:
	Layer(Shape shape, int num_output);

	virtual Matrix<DType>* forward(Matrix<DType>* in) = 0;
	virtual Matrix<DType>* backward(Matrix<DType>* error) = 0;
	virtual void update();

	inline Matrix<DType>* W() { return _weights; }
	inline Shape inShape() { return _in_shape; }
	inline Shape outShape() { return _out_shape; }

protected:
	Matrix<DType>* _in;
	Matrix<DType>* _weights;
	Matrix<DType>* _output;
	Matrix<DType>* _diff;

	Shape _in_shape;
	Shape _out_shape;
};
