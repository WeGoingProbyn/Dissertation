#pragma once

#ifndef INCOMPRESSIBLE_H
#define INCOMPRESSIBLE_H

#include "FVMPhysics.h"

class Incompressible : public Physics {
public:
	void InterimMomentumStep();
	void TrueMomentumStep();
	void SystemDriver();
};
#endif
