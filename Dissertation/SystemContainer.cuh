#pragma once

#ifndef SYSTEMCONTAINER_CUH
#define SYSTEMCONTAINER_CUH

#include <algorithm>
#include <chrono>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <cuda.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "VectorStruct.h"
#include "umfpack.h"

class Container {
public:
	Container();

	__host__ void AllocateSystemMatrixMemoryToCPU();
	__host__ void AllocateInterimMatrixMemoryToCPU();
	__host__ void AllocateSparseIndexMemoryToCPU();
	__host__ void AllocateCompressedRowMemoryToCPU();
	__host__ void AllocateColumnIndexMemoryToCPU();
	__host__ void AllocateLinearSolutionMemoryToCPU();

	__host__ void DeAllocateSystemMatrixMemoryOnCPU();  // CHANGE NAME ON -> TO
	__host__ void DeAllocateInterimMatrixMemoryOnCPU();
	__host__ void DeAllocateSparseIndexMemoryToCPU();
	__host__ void DeAllocateCompressedRowMemoryToCPU();
	__host__ void DeAllocateColumnIndexMemoryToCPU();
	__host__ void DeAllocateLinearSolutionMemoryToCPU();

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

	__host__ double GetDT();
	__host__ double GetNU();
	__host__ double GetRE();
	__host__ vec2 GetD();
	__host__ vec2 GetSIZE();
	__host__ vec2 GetSPLITS();
	__host__ vec4 GetVelocityBoundary();

	__host__ double* GetVariableList();
	__host__ vec3* GetSystemMatrix();
	__host__ vec2* GetInterimMatrix();
	__host__ int* GetSparseIndexI();
	__host__ int* GetSparseIndexJ();
	__host__ int* GetCompressedRowVector();
	__host__ int* GetColumnIndexVector();
	__host__ double* GetRHSVector();
	__host__ double* GetnzCoeffMat();
	__host__ double* GetPSolution();

	__host__ vec3 GetMatrixValue(int i, int j);
	__host__ vec2 GetInterimValue(int i, int j);

	__host__ int SetMatrixValue(int j, int i, double var, const char* dim);
	__host__ int SetInterimValue(int i, int j, double var, const char* dim);
	__host__ int SetLinearValue(int i, int j, double var, const char* dim);
	//__host__ int SetSparseMatrixInfo(int i, int j, const char* dim);
	//__host__ void SetBVector(int i, int j);

	__host__ void BuildSparseMatrixForSolution();
	__host__ void FindSparseLinearSolution();
	__host__ void UpdatePressureValues();

	__host__ bool CheckConvergedExit();
	__host__ bool CheckDivergedExit();
	__host__ void LoopBreakOutput();

	__host__ int ThrowSystemVariables();
	__host__ void CatchSolution();

protected:
	bool debug = true;
	unsigned int SYSTEMSIZE;
	//dim3 DEVICESIZE = dim3(64, 64);
	//dim3 DEVICESPLIT = dim3(32, 32);
	const static unsigned int MAXSPLITS = 4096;
	const static unsigned int MAXSIZE = MAXSPLITS * MAXSPLITS;
	const static unsigned int NUMNONZEROS = (5 * MAXSIZE);

	const static unsigned int VARALLOCSIZE = 10 * sizeof(double);
	const static unsigned int VEC3ALLOCSIZE = MAXSIZE * sizeof(vec3);
	const static unsigned int VEC2ALLOCSIZE = MAXSIZE * sizeof(vec2);
	const static unsigned int DOUBLEALLOCSIZE = MAXSIZE * sizeof(double);
	const static unsigned int DOUBLESPARSEALLOCSIZE = 5 * MAXSIZE * sizeof(double);
	const static unsigned int INTALLOCSIZE = MAXSIZE * sizeof(int);

	int nnz;

	double Control[UMFPACK_CONTROL] = { NULL };
	double Info[UMFPACK_INFO] = { NULL };
	double* null = (double*)NULL;
	void* Symbolic, * Numeric;

private:
	double vars[10];
	vec2 SIZE, SPLIT, D;
	vec4 VelocityBound;
	int STEPNUMBER;
	double RE, NU, RHO, DT, CFL, TOLERANCE;
	double MAXTIME, TRUETIME, BOXSCALE, SIMSTEPS;
	vec2 AverageVelocities, KineticEnergy;
	std::vector<double> VBOUND, DXDY;
	std::vector<double> X, Y, P_SOL;

	vec3* SystemMatrix;
	vec2* InterimMatrix;

	double* RHSVector;
	double* nzCoeffMatrix;     

	int* HostPrefixSum;
	int* SparseIndexesI;
	int* SparseIndexesJ;
	int* RowPointer;
	int* ColumnIndex;        // MAYBE SAVE SPACE BY REUSING SPARSEINDEXESJ
	double* PSolution;
};
#endif