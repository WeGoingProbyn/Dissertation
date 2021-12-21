#pragma once

#ifndef FVMCUDA_CUH
#define FVMCUDA_CUH

#include "SystemContainer.cuh"

class PhysicsCUDA : public Container {
	public:

		__host__ void CudaErrorChecker(cudaError_t func);

		__host__ void AllocateVariablesMemoryToGPU();
		__host__ void AllocateMatrixMemoryToGPU();
		__host__ void AllocateInterimMemoryToGPU();
		__host__ void AllocateLinearSystemMemoryToGPU();

		__host__ void SetVariablesToGPU();
		__host__ void SetMatrixToGPU();
		__host__ void SetInterimToGPU();
		__host__ void SetLinearSystemToGPU();
		__host__ void SetPSolutionToGPU();

		__host__ void GetMatrixFromGPU();
		__host__ void GetInterimFromGPU();
		__host__ void GetLinearSystemFromGPU();

		__host__ void DeviceVariablesCleanUp();
		__host__ void DeviceMatrixCleanUp();
		__host__ void DeviceInterimCleanUp();
		__host__ void DeviceLinearSystemCleanUp();

		//__host__ void ThrowCoefficients();

		using Container::GetMatrixValue;
		__device__ vec3 GetMatrixValue(int i, int j, double* DeviceVars, vec3* DeviceMatrix);
		using Container::GetInterimValue;
		__device__ vec2 GetInterimValue(int i, int j, double* DeviceVars, vec2* DeviceInterim);

		using Container::SetMatrixValue;
		__device__ void SetMatrixValue(int i, int j, double var,  int dim, double* DeviceVars, vec3* DeviceMatrix);
		using Container::SetInterimValue;
		__device__ void SetInterimValue(int i, int j, double var, int dim, double* DeviceVars, vec2* DeviceInterim);
		using Container::SetLinearValue;
		__device__ void SetLinearValue(int i, int j, double var, int dim, double* DeviceVars, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ);

		using Container::GetDT;
		__device__ double GetDT(double* DeviceVars);
		using Container::GetNU;
		__device__ double GetNU(double* DeviceVars);
		using Container::GetSPLITS;
		__device__ vec2 GetSPLITS(double* DeviceVars);
		using Container::GetD;
		__device__ vec2 GetD(double* DeviceVars);
		using Container::GetVelocityBoundary;
		__device__ vec4 GetVelocityBoundary(double* DeviceVars);

		__device__ vec6 InterpolateVelocities(int i, int j, int dim, double* DeviceVars, vec3* DeviceMatrix);
		__device__ double ComputeAdvection(int i, int j, int dim, double* DeviceVars, vec3* DeviceMatrix);
		__device__ double ComputeDiffusion(int i, int j, int dim, double* DeviceVars, vec3* DeviceMatrix);
		__device__ double ComputeMomentum(int i, int j, int dim, double* DeviceVars, vec3* DeviceMatrix);
		__device__ double ComputeIteration(int i, int j, int dim, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim);

		__device__ void SetBaseAValues(int i, int j, double* DeviceVars, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ);
		__device__ void BuildTopLeft(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ);
		__device__ void BuildTopRight(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ);
		__device__ void BuildBottomLeft(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ);
		__device__ void BuildBottomRight(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ);
		__device__ void BuildLeftSide(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ);
		__device__ void BuildRightSide(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ);
		__device__ void BuildTopSide(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ);
		__device__ void BuildBottomSide(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ);
		__device__ void BuildInterior(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ);
		__device__ void BuildLinearSystem(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ);
		
		__device__ void FindValidEntries(int i, int j, double* DeviceVars, double* DeviceNzCoeffMat, int* PrefixSum);
		__device__ void SumValidEntries(int i, int j, double* DeviceVars, double* DeviceNzCoeffMat, int* PrefixSum);
		__device__ void ShiftDeviceLinearSystem(int i, int j, double* DeviceVars, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ, int* PrefixSum);

		using Container::BuildSparseMatrixForSolution;
		__device__ void BuildSparseMatrixForSolution(int i, int j, double* DeviceVars, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ, int* PrefixSum);

	protected:
		double* DeviceVars;
		vec3* DeviceMatrix;
		vec2* DeviceInterim;
		vec3* DeviceMatrixSolution;
		vec2* DeviceInterimSolution;

		double* DeviceRHSVector;
		double* DeviceNzCoeffMat;
		double* DevicePSolution;
		int* DeviceSparseIndexI;
		int* DeviceSparseIndexJ;

		double* DeviceReducedCoeffMat;
		int* DeviceReducedSparseIndexI;
		int* DeviceReducedSparseIndexJ;
		int* PrefixSum;
};

#endif