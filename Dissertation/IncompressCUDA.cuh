#pragma once

#ifndef INCOMPRESSCUDA_CUH
#define INCOMPRESSCUDA_CUH

#include "FVMCUDA.cuh"

class IncompressCUDA : public PhysicsCUDA {
public:
	__host__ void InterimMomentumStep();
	__host__ void LinearSystemBuilder();
	__host__ void TrueMomentumStep();
	__host__ void SystemDriver();
};
#endif
