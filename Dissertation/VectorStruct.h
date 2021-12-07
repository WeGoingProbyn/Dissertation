#pragma once

#ifndef VECTORSTRUCT_H
#define VECTORSTRUCT_H

struct vec2 {
	double x, y;
	vec2() : x(0.0), y(0.0) {}
	vec2(double x, double y) : x(x), y(y) {}
};

struct vec3 {
	double u, v, p;
	vec3() : u(0.0), v(0.0), p(0.0) {}
	__host__ __device__ vec3(double u, double v, double p) : u(u), v(v), p(p) {}
};

struct vec4 {
	double E, W, N, S;
	vec4() : E(0.0), W(0.0), N(0.0), S(0.0) {}
	__host__ __device__ vec4(double E, double W, double N, double S) : E(E), W(W), N(N), S(S) {}
};

struct vec6 {
	double E, W, N, S, EN, WS;
	vec6() : E(0.0), W(0.0), N(0.0), S(0.0), EN(0.0), WS(0.0) {}
	__host__ __device__ vec6(double E1, double W1, double N1, double S1, double N2, double S2)
		: E(E1), W(W1), N(N1), S(S1), EN(N2), WS(S2) {}
};

struct dim6 {
	double Bvec, Acen, Aipos, Aisub, Ajpos, Ajsub;
	dim6() : Bvec(0.0), Acen(0.0), Aipos(0.0), Aisub(0.0), Ajpos(0.0), Ajsub(0.0) {}
	__host__ __device__ dim6(double Bvec, double Acen, double Aipos, double Aisub, double Ajpos, double Ajsub)
		: Bvec(Bvec), Acen(Acen), Aipos(Aipos), Aisub(Aisub), Ajpos(Ajpos), Ajsub(Ajsub) {}
};

#endif
