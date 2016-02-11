#include "../layer.hpp"


template <typename DType>
Layer<DType>::LayerIterator::LayerIterator(Layer<DType>* layer)
{
	_current = layer;
	_previous = nullptr;
}

template <typename DType>
inline void Layer<DType>::LayerIterator::operator++()
{
	for (Layer<DType>* layer : _current->_next)
	{
		_queue.push(std::make_pair(_current, layer));
	}

	if (_queue.empty())
	{
		_current = nullptr;
	}
	else
	{
		auto pair = _queue.front();
		_previous = pair.first;
		_current = pair.second;
		_queue.pop();
	}
}

template <typename DType>
inline void Layer<DType>::LayerIterator::operator--()
{
	for (Layer<DType>* layer : _current->_previous)
	{
		_queue.push(std::make_pair(_current, layer));
	}

	if (_queue.empty())
	{
		_current = nullptr;
	}
	else
	{
		auto pair = _queue.front();
		_previous = pair.first;
		_current = pair.second;
		_queue.pop();
	}
}

template <typename DType>
inline Layer<DType>* Layer<DType>::LayerIterator::operator*()
{
	return _current;
}

template <typename DType>
inline bool Layer<DType>::LayerIterator::next()
{
	return _current != nullptr;
}

template <typename DType>
inline Layer<DType>* Layer<DType>::LayerIterator::last()
{
	return _previous;
}


// Specialization
template Layer<float>::LayerIterator::LayerIterator(Layer<float>* layer);
template Layer<double>::LayerIterator::LayerIterator(Layer<double>* layer);

template void Layer<float>::LayerIterator::operator++();
template void Layer<double>::LayerIterator::operator++();

template void Layer<float>::LayerIterator::operator--();
template void Layer<double>::LayerIterator::operator--();

template Layer<float>* Layer<float>::LayerIterator::operator*();
template Layer<double>* Layer<double>::LayerIterator::operator*();

template bool Layer<float>::LayerIterator::next();
template bool Layer<double>::LayerIterator::next();

template Layer<float>* Layer<float>::LayerIterator::last();
template Layer<double>* Layer<double>::LayerIterator::last();
