#pragma once

#ifndef INCOMPRESSCUDA_H
#define INCOMPRESSCUDA_H

#include "FVMCUDA.cuh"

class IncompressCUDA : public PhysicsCUDA {
public:
	__host__ void InterimMomentumStep();
	__host__ void TrueMomentumStep();
	__host__ void SystemDriver();
};
#endif
