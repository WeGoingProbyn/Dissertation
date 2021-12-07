#pragma once

#ifndef FVMPHYSICS_H
#define FVMPHYSICS_H

#include <limits.h>
#include "SystemContainer.cuh"

class Physics : public Container {
public:
	vec6 InterpolateVelocities(int i, int j, const char* dim);
	double ComputeAdvection(int i, int j, const char* dim);
	double ComputeDiffusion(int i, int j, const char* dim);
	void ComputeMomentum(int i, int j, const char* dim);

	void BuildCoeffMat();
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
	void ReshapeCoefficients();
	void BuildLinearSystem();

	void ComputeIteration(int i, int j, const char* dim);

	void ThrowCoefficients();

private:
	std::vector<std::vector<dim6>> LinearSystemMatrix;
	std::vector<dim6> LinearSystemRESHAPED;
};

#endif