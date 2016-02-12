#pragma once

struct Shape
{
	int m;
	int n;

	Shape() : Shape(0, 0) {}
	Shape(int m, int n) : m(m), n(n) {}
	inline int prod() { return m*n; }
	inline Shape T() { return{ n, m }; }
};
