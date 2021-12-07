#pragma once

#ifndef FVMCUDA_H
#define FVMCUDA_H

#include "SystemContainer.cuh"

class PhysicsCUDA : public Container {
	public:
		//__host__ PhysicsCUDA GetDeviceObject();

		__host__ void LinearSystemGPUWrapper();

		__host__ void CudaErrorChecker(cudaError_t func);

		//__host__ void AllocateObjectMemoryToGPU();
		__host__ void AllocateMatrixMemoryToGPU();
		__host__ void AllocateMatrixSolutionToGPU();
		__host__ void AllocateInterimMemoryToGPU();
		__host__ void AllocateInterimSolutionToGPU();
		__host__ void AllocateLinearSystemMemoryToGPU();

		//__host__ void SetObjectToGPU();
		__host__ void SetMatrixToGPU();
		__host__ void SetInterimToGPU();
		__host__ void SetLinearSystemToGPU();
		__host__ void GetMatrixFromGPU();
		__host__ void GetInterimFromGPU();
		__host__ void GetLinearSystemFromGPU();

		//__host__ void DeviceObjectCleanUp();
		__host__ void DeviceMatrixCleanUp();
		__host__ void DeviceInterimCleanUp();
		__host__ void DeviceLinearSystemCleanUp();

		using Container::GetMatrixValue;
		__device__ vec3 GetMatrixValue(int i, int j);
		using Container::GetInterimValue;
		__device__ vec2 GetInterimValue(int i, int j);

		using Container::SetMatrixValue;
		__device__ void SetMatrixValue(int i, int j, double var, const char* dim);
		using Container::SetInterimValue;
		__device__ void SetInterimValue(int i, int j, double var, const char* dim);

		__device__ vec6 InterpolateVelocities(int i, int j, const char* dim);
		__device__ double ComputeAdvection(int i, int j, const char* dim);
		__device__ double ComputeDiffusion(int i, int j, const char* dim);
		__device__ void ComputeMomentum(int i, int j, const char* dim);

		__device__ void SetBaseAValues(int i, int j);
		__device__ void BuildTopLeft(int i, int j);
		__device__ void BuildTopRight(int i, int j);
		__device__ void BuildBottomLeft(int i, int j);
		__device__ void BuildBottomRight(int i, int j);
		__device__ void BuildLeftSide(int i, int j);
		__device__ void BuildRightSide(int i, int j);
		__device__ void BuildTopSide(int i, int j);
		__device__ void BuildBottomSide(int i, int j);
		__device__ void BuildInterior(int i, int j);
		__device__ void BuildLinearSystem(int i, int j);

		__device__ void ComputeIteration(int i, int j, const char* dim);

		__host__ void BuildCoeffMat();
		__host__ void ReshapeCoefficients();
		__host__ void ThrowCoefficients();

	private:
		//PhysicsCUDA* DeviceObject;

		vec3** DeviceMatrix;
		vec2** DeviceInterim;
		vec3** DeviceMatrixSolution;
		vec2** DeviceInterimSolution;

		dim6** DeviceLinearSystem;
		
		const int MAXSPLITS = 4096;
		const int MAXSIZE = 4096 * 4096;
		int OBJECTSIZE = sizeof(PhysicsCUDA);
		int SYSTEMSIZE = (int)GetSPLITS().x + 1 * (int)GetSPLITS().y + 1;
		const int VEC3ALLOCSIZE = MAXSIZE* sizeof(vec3);
		const int VEC2ALLOCSIZE = MAXSIZE * sizeof(vec2);
		const int DIM6ALLOCSIZE = MAXSIZE * sizeof(dim6);
		std::vector<std::vector<dim6>> LinearSystemMatrix;
		std::vector<dim6> LinearSystemRESHAPED;
};

#endif