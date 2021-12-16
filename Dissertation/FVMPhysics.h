#pragma once

#ifndef FVMPHYSICS_H
#define FVMPHYSICS_H

#include "SystemContainer.cuh"

class Physics : public Container {
public:
	vec6 InterpolateVelocities(int i, int j, const char* dim);
	double ComputeAdvection(int i, int j, const char* dim);
	double ComputeDiffusion(int i, int j, const char* dim);
	void ComputeMomentum(int i, int j, const char* dim);

	void SetBaseAValues();
	void BuildTopLeft();
	void BuildTopRight();
	void BuildBottomLeft();
	void BuildBottomRight();
	void BuildLeftSide();
	void BuildRightSide();
	void BuildTopSide();
	void BuildBottomSide();
	void BuildInterior();
	void BuildLinearSystem();

	void ComputeIteration(int i, int j, const char* dim);

	//void ThrowCoefficients();
};

#endif