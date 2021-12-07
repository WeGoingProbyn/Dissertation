#pragma once

#ifndef SYSTEMCONTAINER_H
#define SYSTEMCONTAINER_H

#include <algorithm>
#include <chrono>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "VectorStruct.h"

class Container {
	public:
		Container();



		__host__ int SetRE(double re);
		__host__ int SetCFL(double cfl);
		__host__ int SetVelocityBoundary(vec4 vbound);
		__host__ int SetSPLITS(vec2 SPLITS);
		__host__ int SetSIZE(vec2 SIZES);
		__host__ int SetMAXTIME(double TIME);
		__host__ int SetSystemVariables();
		__host__ void SetTOLERANCE(double TOL);
		__host__ void SetAverageVelocities();
		__host__ void SetKineticEnergy();

		__host__ double GetCFL();
		__host__ double GetMAXTIME();
		__host__ double GetTOLERANCE();
		__host__ int GetSystemVariables();
		__host__ int GetSIMSTEPS();
		__host__ int GetCURRENTSTEP();
		__host__ int IncreaseSTEP();
		__host__ vec2 GetKineticEnergy();

		__host__ __device__ double GetDT();
		__host__ __device__ double GetNU();
		__host__ __device__ double GetRE();
		__host__ __device__ vec2 GetD();
		__host__ __device__ vec2 GetSIZE();
		__host__ __device__ vec2 GetSPLITS();
		__host__ __device__ vec4 GetVelocityBoundary();

		__host__ std::vector<std::vector<vec3>> GetSystemMatrix();
		__host__ std::vector<std::vector<vec2>> GetInterimMatrix();

		__host__ vec3 GetMatrixValue(int i, int j);
		__host__ vec2 GetInterimValue(int i, int j);

		__host__ int SetMatrixValue(int j, int i, double var, const char *dim);
		__host__ int SetInterimValue(int i, int j, double var, const char *dim);

		__host__ bool CheckConvergedExit();
		__host__ bool CheckDivergedExit();
		__host__ void LoopBreakOutput();

		__host__ int ThrowSystemVariables();
		__host__ void CatchSolution();

	private:
		vec2 SIZE, SPLIT, D;
		vec4 VelocityBound;
		int STEPNUMBER;
		double RE, NU, RHO, DT, CFL, TOLERANCE;
		double MAXTIME, TRUETIME, BOXSCALE, SIMSTEPS;
		vec2 AverageVelocities, KineticEnergy;
		std::vector<double> VBOUND, DXDY;
		std::vector<double> X, Y, P_SOL;
		std::vector<std::vector<vec3>> SystemMatrix;
		std::vector<std::vector<vec2>> InterimMatrix;
	};
#endif