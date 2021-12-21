#include "FVMCUDA.cuh"

__host__ void PhysicsCUDA::CudaErrorChecker(cudaError_t func) {
    if (debug) {
        if (func == cudaSuccess) { std::cout << " CudaSuccess" << std::endl; }
        else { std::cout << stderr << " cudaFailure: " << cudaGetErrorString(func) << std::endl; }
    }
    else { 
        if (func == cudaSuccess) { ; }
        else { std::cout << stderr << " cudaFailure: " << cudaGetErrorString(func) << std::endl; }
    }
}

__host__ void PhysicsCUDA::AllocateVariablesMemoryToGPU() {
    if(debug) { std::cout << "AllocateVariablesMemoryToGPU ->"; }
    CudaErrorChecker(cudaMalloc(&DeviceVars, VARALLOCSIZE));
    if (debug) { std::cout << "=====================" << std::endl; }
}

__host__ void PhysicsCUDA::AllocateMatrixMemoryToGPU() {
    if (debug) { std::cout << "AllocateMatrixMemoryToGPU ->"; }
    CudaErrorChecker(cudaMalloc(&DeviceMatrix, VEC3ALLOCSIZE));
    if (debug) { std::cout << "=====================" << std::endl; }
}

__host__ void PhysicsCUDA::AllocateInterimMemoryToGPU() {
    if (debug) { std::cout << "AllocateInterimMemoryToGPU ->"; }
    CudaErrorChecker(cudaMalloc(&DeviceInterim, VEC2ALLOCSIZE));
    if (debug) { std::cout << "=====================" << std::endl; }
}

__host__ void PhysicsCUDA::AllocateLinearSystemMemoryToGPU() {
    if (debug) { std::cout << "AllocateLinearSystemMemoryToGPU ->\n" << "DeviceRHSVector ->"; }
    CudaErrorChecker(cudaMalloc(&DeviceRHSVector, DOUBLEALLOCSIZE));
    if (debug) { std::cout << "DeviceNzCoeffMat ->"; }
    CudaErrorChecker(cudaMalloc(&DeviceNzCoeffMat, DOUBLESPARSEALLOCSIZE));
    if (debug) { std::cout << "DevicePSolution ->"; }
    CudaErrorChecker(cudaMalloc(&DevicePSolution, DOUBLEALLOCSIZE));
    if (debug) { std::cout << "DeviceSparseIndexI ->"; }
    CudaErrorChecker(cudaMalloc(&DeviceSparseIndexI, INTALLOCSIZE));
    if (debug) { std::cout << "DeviceSparseIndexJ ->"; }
    CudaErrorChecker(cudaMalloc(&DeviceSparseIndexJ, INTALLOCSIZE));
    if (debug) { std::cout << "PrefixSum ->"; }
    CudaErrorChecker(cudaMalloc(&PrefixSum, INTALLOCSIZE));
    if (debug) { std::cout << "=====================" << std::endl; }
}

__host__ void PhysicsCUDA::SetVariablesToGPU() {
    if (debug) { std::cout << "SetVariablesToGPU ->"; }
    CudaErrorChecker(cudaMemcpy(DeviceVars, GetVariableList(), VARALLOCSIZE, cudaMemcpyHostToDevice));
    if (debug) { std::cout << "=====================" << std::endl; }
}

__host__ void PhysicsCUDA::SetMatrixToGPU() {
    if (debug) { std::cout << "SetMatrixToGPU ->"; }
    CudaErrorChecker(cudaMemcpy(DeviceMatrix, GetSystemMatrix(), VEC3ALLOCSIZE, cudaMemcpyHostToDevice));
    if (debug) { std::cout << "=====================" << std::endl; }
}

__host__ void PhysicsCUDA::SetInterimToGPU() {
    if (debug) { std::cout << "SetInterimToGPU ->"; }
    CudaErrorChecker(cudaMemcpy(DeviceInterim, GetInterimMatrix(), VEC2ALLOCSIZE, cudaMemcpyHostToDevice));
    if (debug) { std::cout << "=====================" << std::endl; }
}

__host__ void PhysicsCUDA::SetLinearSystemToGPU() {
    if (debug) { std::cout << "SetLinearSystemToGPU ->\n" << "DeviceRHSVector ->"; }
    CudaErrorChecker(cudaMemcpy(DeviceRHSVector, GetRHSVector(), DOUBLEALLOCSIZE, cudaMemcpyHostToDevice));
    if (debug) { std::cout << "DeviceNzCoeffMat ->"; }
    CudaErrorChecker(cudaMemcpy(DeviceNzCoeffMat, GetnzCoeffMat(), DOUBLESPARSEALLOCSIZE, cudaMemcpyHostToDevice));
    if (debug) { std::cout << "DeviceSparseIndexI ->"; }
    CudaErrorChecker(cudaMemcpy(DeviceSparseIndexI, GetSparseIndexI(), INTALLOCSIZE, cudaMemcpyHostToDevice));
    if (debug) { std::cout << "DeviceSparseIndexJ ->"; }
    CudaErrorChecker(cudaMemcpy(DeviceSparseIndexJ, GetSparseIndexJ(), INTALLOCSIZE, cudaMemcpyHostToDevice));
    if (debug) { std::cout << "=====================" << std::endl; }
}

__host__ void PhysicsCUDA::SetPSolutionToGPU() {
    if (debug) { std::cout << "DevicePSolution ->"; }
    CudaErrorChecker(cudaMemcpy(DevicePSolution, GetPSolution(), DOUBLEALLOCSIZE, cudaMemcpyHostToDevice));
    if (debug) { std::cout << "=====================" << std::endl; }
}

__host__ void PhysicsCUDA::GetMatrixFromGPU() {
    if (debug) { std::cout << "GetMatrixFromGPU ->"; }
    CudaErrorChecker(cudaMemcpy(GetSystemMatrix(), DeviceMatrix, VEC3ALLOCSIZE, cudaMemcpyDeviceToHost));
    if (debug) { std::cout << "=====================" << std::endl; }
}

__host__ void PhysicsCUDA::GetInterimFromGPU() {
    if (debug) { std::cout << "GetInterimFromGPU ->"; }
    CudaErrorChecker(cudaMemcpy(GetInterimMatrix(), DeviceInterim, VEC2ALLOCSIZE, cudaMemcpyDeviceToHost));
    if (debug) { std::cout << "=====================" << std::endl; }
}

__host__ void PhysicsCUDA::GetLinearSystemFromGPU() {
    if (debug) { std::cout << "GetLinearSystemFromGPU ->\n" << "DeviceRHSVector ->"; }
    CudaErrorChecker(cudaMemcpy(GetRHSVector(), DeviceRHSVector, DOUBLEALLOCSIZE, cudaMemcpyDeviceToHost));
    if (debug) { std::cout << "DeviceNzCoeffMat ->"; }
    CudaErrorChecker(cudaMemcpy(GetnzCoeffMat(), DeviceNzCoeffMat, DOUBLESPARSEALLOCSIZE, cudaMemcpyDeviceToHost));
    if (debug) { std::cout << "DeviceSparseIndexI ->"; }
    CudaErrorChecker(cudaMemcpy(GetSparseIndexI(), DeviceSparseIndexI, INTALLOCSIZE, cudaMemcpyDeviceToHost));
    if (debug) { std::cout << "DeviceSparseIndexJ ->"; }
    CudaErrorChecker(cudaMemcpy(GetSparseIndexJ(), DeviceSparseIndexJ, INTALLOCSIZE, cudaMemcpyDeviceToHost));
    if (debug) { std::cout << "=====================" << std::endl; }
}

__host__ void PhysicsCUDA::DeviceVariablesCleanUp() { cudaFree(DeviceVars); }

__host__ void PhysicsCUDA::DeviceMatrixCleanUp() { cudaFree(DeviceMatrix); }

__host__ void PhysicsCUDA::DeviceInterimCleanUp() { cudaFree(DeviceInterim); }

__host__ void PhysicsCUDA::DeviceLinearSystemCleanUp() { 
    cudaFree(DeviceRHSVector); 
    cudaFree(DeviceNzCoeffMat); 
    cudaFree(DevicePSolution); 
    cudaFree(DeviceSparseIndexI); 
    cudaFree(DeviceSparseIndexJ); 
    cudaFree(PrefixSum);
}

__device__ vec2 PhysicsCUDA::GetSPLITS(double* DeviceVars) { return vec2(DeviceVars[0], DeviceVars[1]); }

__device__ vec2 PhysicsCUDA::GetD(double* DeviceVars) { return vec2(DeviceVars[2], DeviceVars[3]); }

__device__ double PhysicsCUDA::GetDT(double* DeviceVars) { return DeviceVars[4]; }

__device__ double PhysicsCUDA::GetNU(double* DeviceVars) { return DeviceVars[5]; }

__device__ vec4 PhysicsCUDA::GetVelocityBoundary(double* DeviceVars) { return vec4(DeviceVars[6], DeviceVars[7], DeviceVars[8], DeviceVars[9]); }

__device__ vec3 PhysicsCUDA::GetMatrixValue(int i, int j, double* DeviceVars, vec3* DeviceMatrix) {
    int index = (j * (int)GetSPLITS(DeviceVars).y) + i;
    return DeviceMatrix[index];
}

__device__ vec2 PhysicsCUDA::GetInterimValue(int i, int j, double* DeviceVars, vec2* DeviceInterim) {
    int index = (j * (int)GetSPLITS(DeviceVars).y) + i;
    return DeviceInterim[index];
}

__device__ void PhysicsCUDA::SetMatrixValue(int i, int j, double var, int dim, double* DeviceVars, vec3* DeviceMatrix) {
    int index = (j * (int)GetSPLITS(DeviceVars).y) + i;
    if (dim == 0) { DeviceMatrix[index].u = var; }
    else if (dim == 1) { DeviceMatrix[index].v = var; }
    else if (dim == 2) { DeviceMatrix[index].p = var; }
}

__device__ void PhysicsCUDA::SetInterimValue(int i, int j, double var, int dim, double* DeviceVars, vec2* DeviceInterim) {
    int index = (j * (int)GetSPLITS(DeviceVars).y) + i;
    if (dim == 0) { DeviceInterim[index].x = var; }
    else if (dim == 1) { DeviceInterim[index].y = var; }
}

__device__ void PhysicsCUDA::SetLinearValue(int i, int j, double var, int dim, double* DeviceVars, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ) {
    int TRUEindex = (j * (int)GetSPLITS(DeviceVars).y) + i;
    if (dim == 0) { DeviceRHSVector[TRUEindex] = var; }
    if (dim == 1) {
        DeviceSparseIndexI[TRUEindex] = TRUEindex;
        DeviceSparseIndexJ[TRUEindex] = TRUEindex;
        DeviceNzCoeffMat[TRUEindex] = var;
    }
    if (dim == 2) { 
        int index = (j * GetSPLITS(DeviceVars).y) + i + (GetSPLITS(DeviceVars).y * GetSPLITS(DeviceVars).x);
        DeviceSparseIndexI[index] = TRUEindex + 1;
        DeviceSparseIndexJ[index] = TRUEindex;
        DeviceNzCoeffMat[index] = var;
    }
    if (dim == 3) {
        int index = (j * GetSPLITS(DeviceVars).y) + i + (2 * (GetSPLITS(DeviceVars).y * GetSPLITS(DeviceVars).x)); 
        DeviceSparseIndexI[index] = TRUEindex - 1;
        DeviceSparseIndexJ[index] = TRUEindex;
        DeviceNzCoeffMat[index] = var;  
    }
    if (dim == 4) {
        int index = (j * GetSPLITS(DeviceVars).y) + i + (3 * (GetSPLITS(DeviceVars).y * GetSPLITS(DeviceVars).x)); 
        DeviceSparseIndexI[index] = TRUEindex;
        DeviceSparseIndexJ[index] = TRUEindex - GetSPLITS(DeviceVars).y;
        DeviceNzCoeffMat[index] = var; 
    }
    if (dim == 5) {
        int index = (j * GetSPLITS(DeviceVars).y) + i + (4 * (GetSPLITS(DeviceVars).y * GetSPLITS(DeviceVars).x));
        DeviceSparseIndexI[index] = TRUEindex;
        DeviceSparseIndexJ[index] = TRUEindex + GetSPLITS(DeviceVars).y;
        DeviceNzCoeffMat[index] = var;  
    }
}

__device__ vec6 PhysicsCUDA::InterpolateVelocities(int i, int j, int dim, double* DeviceVars, vec3* DeviceMatrix) {
    if (dim == 0) {
        double UEAST = 0.5 * (GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).u + GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u);
        double UWEST = 0.5 * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u + GetMatrixValue(i - 1, j, DeviceVars, DeviceMatrix).u);
        double VNORTH = 0.5 * (GetMatrixValue(i - 1, j + 1, DeviceVars, DeviceMatrix).v + GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).v);
        double VSOUTH = 0.5 * (GetMatrixValue(i - 1, j, DeviceVars, DeviceMatrix).v + GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v);

        if (j == 0) {
            double UNORTH = 0.5 * (GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).u + GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u);
            double USOUTH = GetVelocityBoundary(DeviceVars).E;
            return vec6(UEAST, UWEST, UNORTH, USOUTH, VNORTH, VSOUTH);
        }
        else if (j == GetSPLITS(DeviceVars).y - 1) {
            double UNORTH = GetVelocityBoundary(DeviceVars).W;
            double USOUTH = 0.5 * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u + GetMatrixValue(i, j - 1, DeviceVars, DeviceMatrix).u);
            return vec6(UEAST, UWEST, UNORTH, USOUTH, VNORTH, VSOUTH);
        }
        else {
            double UNORTH = 0.5 * (GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).u + GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u);
            double USOUTH = 0.5 * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u + GetMatrixValue(i, j - 1, DeviceVars, DeviceMatrix).u);
            return vec6(UEAST, UWEST, UNORTH, USOUTH, VNORTH, VSOUTH);
        }
    }
    else if (dim == 1) {
        double VNORTH = 0.5 * (GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).v + GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v);
        double VSOUTH = 0.5 * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v + GetMatrixValue(i, j - 1, DeviceVars, DeviceMatrix).v);
        double UEAST = 0.5 * (GetMatrixValue(i + 1, j - 1, DeviceVars, DeviceMatrix).u + GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).u);
        double UWEST = 0.5 * (GetMatrixValue(i, j - 1, DeviceVars, DeviceMatrix).u + GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u);

        if (i == 0) {
            double VEAST = 0.5 * (GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).v + GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v);
            double VWEST = GetVelocityBoundary(DeviceVars).S;
            return vec6(VEAST, VWEST, VNORTH, VSOUTH, UEAST, UWEST);
        }
        else if (i == GetSPLITS(DeviceVars).x - 1) {
            double VEAST = GetVelocityBoundary(DeviceVars).N;
            double VWEST = 0.5 * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v + GetMatrixValue(i - 1, j, DeviceVars, DeviceMatrix).v);
            return vec6(VEAST, VWEST, VNORTH, VSOUTH, UEAST, UWEST);
        }
        else {
            double VEAST = 0.5 * (GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).v + GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v);
            double VWEST = 0.5 * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v + GetMatrixValue(i - 1, j, DeviceVars, DeviceMatrix).v);
            return vec6(VEAST, VWEST, VNORTH, VSOUTH, UEAST, UWEST);
        }
    }
    //    else {
    //        std::cout << "No dimension found" << std::endl;
    //        throw - 1;
    //    }
}

__device__ double PhysicsCUDA::ComputeAdvection(int i, int j, int dim, double* DeviceVars, vec3* DeviceMatrix) {
    if (dim == 0) {
        vec6 var1 = InterpolateVelocities(i, j, dim, DeviceVars, DeviceMatrix);
        double XX = (var1.E * var1.E - var1.W * var1.W) / GetD(DeviceVars).x;
        double XY = (var1.N * var1.EN - var1.S * var1.WS) / GetD(DeviceVars).y;
        return -(XX + XY);
    }

    else if (dim == 1) {
        vec6 var2 = InterpolateVelocities(i, j, dim, DeviceVars, DeviceMatrix);
        double YY = (var2.N * var2.N - var2.S * var2.S) / GetD(DeviceVars).y;
        double YX = (var2.E * var2.EN - var2.W * var2.WS) / GetD(DeviceVars).x;
        return -(YY + YX);
    }
}

__device__ double PhysicsCUDA::ComputeDiffusion(int i, int j, int dim, double* DeviceVars, vec3* DeviceMatrix) {
    if (dim == 0) {
        double XDXe = -2 * GetNU(DeviceVars) * (GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).u - GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u) / GetD(DeviceVars).x;
        double XDXw = -2 * GetNU(DeviceVars) * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u - GetMatrixValue(i - 1, j, DeviceVars, DeviceMatrix).u) / GetD(DeviceVars).x;

        if (j == GetSPLITS(DeviceVars).y - 1) {
            double XDYn = -GetNU(DeviceVars) * (GetVelocityBoundary(DeviceVars).E - GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u) / (GetD(DeviceVars).y / 2) -
                           GetNU(DeviceVars) * (GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).v - GetMatrixValue(i - 1, j + 1, DeviceVars, DeviceMatrix).v) / GetD(DeviceVars).x;
            double XDYs = -GetNU(DeviceVars) * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u - GetMatrixValue(i, j - 1, DeviceVars, DeviceMatrix).u) / GetD(DeviceVars).y -
                           GetNU(DeviceVars) * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v - GetMatrixValue(i - 1, j, DeviceVars, DeviceMatrix).v) / GetD(DeviceVars).x;
            return (XDXe - XDXw) / GetD(DeviceVars).x + (XDYn - XDYs) / GetD(DeviceVars).y;
        }
        if (j == 0) {
            double XDYn = -GetNU(DeviceVars) * (GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).u - GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u) / GetD(DeviceVars).y -
                           GetNU(DeviceVars) * (GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).v - GetMatrixValue(i - 1, j + 1, DeviceVars, DeviceMatrix).v) / GetD(DeviceVars).x;
            double XDYs = -GetNU(DeviceVars) * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u - GetVelocityBoundary(DeviceVars).W) / (GetD(DeviceVars).y / 2) -
                           GetNU(DeviceVars) * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v - GetMatrixValue(i - 1, j, DeviceVars, DeviceMatrix).v) / GetD(DeviceVars).x;
            return (XDXe - XDXw) / GetD(DeviceVars).x + (XDYn - XDYs) / GetD(DeviceVars).y;
        }
        else {
            double XDYn = -GetNU(DeviceVars) * (GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).u - GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u) / GetD(DeviceVars).y -
                           GetNU(DeviceVars) * (GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).v - GetMatrixValue(i - 1, j + 1, DeviceVars, DeviceMatrix).v) / GetD(DeviceVars).x;
            double XDYs = -GetNU(DeviceVars) * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u - GetMatrixValue(i, j - 1, DeviceVars, DeviceMatrix).u) / GetD(DeviceVars).y -
                          GetNU(DeviceVars) * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v - GetMatrixValue(i - 1, j, DeviceVars, DeviceMatrix).v) / GetD(DeviceVars).x;
            return (XDXe - XDXw) / GetD(DeviceVars).x + (XDYn - XDYs) / GetD(DeviceVars).y;
        }
    }
    else if (dim == 1) {
        double YDYn = -2 * GetNU(DeviceVars) * (GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).v - GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v) / GetD(DeviceVars).y;
        double YDYs = -2 * GetNU(DeviceVars) * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v - GetMatrixValue(i, j - 1, DeviceVars, DeviceMatrix).v) / GetD(DeviceVars).y;

        if (i == GetSPLITS(DeviceVars).x - 1) {
            double YDXe = -GetNU(DeviceVars) * (GetVelocityBoundary(DeviceVars).S - GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v) / (GetD(DeviceVars).x / 2) -
                           GetNU(DeviceVars) * (GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).u - GetMatrixValue(i + 1, j - 1, DeviceVars, DeviceMatrix).u) / GetD(DeviceVars).y;
            double YDXw = -GetNU(DeviceVars) * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v - GetMatrixValue(i - 1, j, DeviceVars, DeviceMatrix).v) / GetD(DeviceVars).x -
                           GetNU(DeviceVars) * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u - GetMatrixValue(i, j - 1, DeviceVars, DeviceMatrix).u) / GetD(DeviceVars).y;
            return (YDYn - YDYs) / GetD(DeviceVars).y + (YDXe - YDXw) / GetD(DeviceVars).x;
        }
        if (i == 0) {
            double YDXe = -GetNU(DeviceVars) * (GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).v - GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v) / GetD(DeviceVars).x -
                           GetNU(DeviceVars) * (GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).u - GetMatrixValue(i + 1, j - 1, DeviceVars, DeviceMatrix).u) / GetD(DeviceVars).y;
            double YDXw = -GetNU(DeviceVars) * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v - GetVelocityBoundary(DeviceVars).N) / (GetD(DeviceVars).x / 2) -
                           GetNU(DeviceVars) * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u - GetMatrixValue(i, j - 1, DeviceVars, DeviceMatrix).u) / GetD(DeviceVars).y;
            return (YDYn - YDYs) / GetD(DeviceVars).y + (YDXe - YDXw) / GetD(DeviceVars).x;
        }
        else {
            double YDXe = -GetNU(DeviceVars) * (GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).v - GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v) / GetD(DeviceVars).x -
                           GetNU(DeviceVars) * (GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).u - GetMatrixValue(i + 1, j - 1, DeviceVars, DeviceMatrix).u) / GetD(DeviceVars).y;
            double YDXw = -GetNU(DeviceVars) * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v - GetMatrixValue(i - 1, j, DeviceVars, DeviceMatrix).v) / GetD(DeviceVars).x -
                           GetNU(DeviceVars) * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u - GetMatrixValue(i, j - 1, DeviceVars, DeviceMatrix).u) / GetD(DeviceVars).y;
            return (YDYn - YDYs) / GetD(DeviceVars).y + (YDXe - YDXw) / GetD(DeviceVars).x;
        }
    }
}

__device__ double PhysicsCUDA::ComputeMomentum(int i, int j, int dim, double* DeviceVars, vec3* DeviceMatrix) {
    double var = (ComputeAdvection(i, j, dim, DeviceVars, DeviceMatrix) - ComputeDiffusion(i, j, dim, DeviceVars, DeviceMatrix));
    return var;
}

__device__ void PhysicsCUDA::SetBaseAValues(int i, int j, double* DeviceVars, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ) {
    double var = -GetDT(DeviceVars) * (GetD(DeviceVars).y / GetD(DeviceVars).x);
    if (i < GetSPLITS(DeviceVars).x && j < GetSPLITS(DeviceVars).y) {
        SetLinearValue(i, j, var, 2, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        SetLinearValue(i, j, var, 3, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        SetLinearValue(i, j, var, 4, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        SetLinearValue(i, j, var, 5, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
    }
}

__device__ void PhysicsCUDA::BuildTopLeft(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ) {
    if (i == 0 && j == 0) {
        double var;
        var = -GetD(DeviceVars).y * (GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).u + GetDT(DeviceVars) * GetInterimValue(i + 1, j, DeviceVars, DeviceInterim).x) +
               GetD(DeviceVars).y * GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u -
               GetD(DeviceVars).x * (GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).v + GetDT(DeviceVars) * GetInterimValue(i, j + 1, DeviceVars, DeviceInterim).y) +
               GetD(DeviceVars).x * GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v;
        SetLinearValue(i, j, var, 0, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        var = (GetDT(DeviceVars) * (GetD(DeviceVars).y / GetD(DeviceVars).x)) + (GetDT(DeviceVars) * (GetD(DeviceVars).x / GetD(DeviceVars).y));
        SetLinearValue(i, j, var, 1, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        SetLinearValue(i, j, 0.0, 3, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ); // No slip
        SetLinearValue(i, j, 0.0, 4, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ); // No slip
    }
}

__device__ void PhysicsCUDA::BuildTopRight(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ) { // Bottom Left
    double var;
    if (i == 0 && j == (int)GetSPLITS(DeviceVars).y - 1) {
        var = -GetD(DeviceVars).y * (GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).u + GetDT(DeviceVars) * GetInterimValue(i + 1, j, DeviceVars, DeviceInterim).x) +
               GetD(DeviceVars).y * GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u -
               GetD(DeviceVars).x * GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).v +
               GetD(DeviceVars).x * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v + GetDT(DeviceVars) * GetInterimValue(i, j, DeviceVars, DeviceInterim).y);
        SetLinearValue(i, j, var, 0, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        var = (GetDT(DeviceVars) * (GetD(DeviceVars).y / GetD(DeviceVars).x)) + GetDT(DeviceVars) * (GetD(DeviceVars).x / GetD(DeviceVars).y);
        SetLinearValue(i, j, var, 1, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        SetLinearValue(i, j, 0.0, 3, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        SetLinearValue(i, j, 0.0, 5, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
    }
}

__device__ void PhysicsCUDA::BuildBottomLeft(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ) { // Top Right
    double var;
    if (i == (int)GetSPLITS(DeviceVars).x - 1 && j == 0) {
        var = -GetD(DeviceVars).y * GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).u +
               GetD(DeviceVars).y * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u + GetDT(DeviceVars) * GetInterimValue(i, j, DeviceVars, DeviceInterim).x) -
               GetD(DeviceVars).x * (GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).v + GetDT(DeviceVars) * GetInterimValue(i, j + 1, DeviceVars, DeviceInterim).y) +
               GetD(DeviceVars).x * GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v;
        SetLinearValue(i, j, var, 0, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        var = (GetDT(DeviceVars) * (GetD(DeviceVars).y / GetD(DeviceVars).x)) + (GetDT(DeviceVars) * (GetD(DeviceVars).x / GetD(DeviceVars).y));
        SetLinearValue(i, j, var, 1, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        SetLinearValue(i, j, 0.0, 2, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        SetLinearValue(i, j, 0.0, 4, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
    }
}

__device__ void PhysicsCUDA::BuildBottomRight(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ) {
    double var;
    if (i == (int)GetSPLITS(DeviceVars).x - 1 && j == (int)GetSPLITS(DeviceVars).y - 1) {
        var = -GetD(DeviceVars).y * GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).u +
               GetD(DeviceVars).y * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u + GetDT(DeviceVars) * GetInterimValue(i, j, DeviceVars, DeviceInterim).x) -
               GetD(DeviceVars).x * GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).v +
               GetD(DeviceVars).x * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v + GetDT(DeviceVars) * GetInterimValue(i, j, DeviceVars, DeviceInterim).y);
        SetLinearValue(i, j, var, 0, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        var = (GetDT(DeviceVars) * (GetD(DeviceVars).y / GetD(DeviceVars).x)) + (GetDT(DeviceVars) * (GetD(DeviceVars).x / GetD(DeviceVars).y));
        SetLinearValue(i, j, var, 1, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        SetLinearValue(i, j, 0.0, 2, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        SetLinearValue(i, j, 0.0, 5, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
    }
}

__device__ void PhysicsCUDA::BuildLeftSide(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ) { // Top side
    double var;
    if (j == 0) {
        if (i > 0 && i < GetSPLITS(DeviceVars).x - 1) {
            var = -GetD(DeviceVars).y * (GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).u + GetDT(DeviceVars) * GetInterimValue(i + 1, j, DeviceVars, DeviceInterim).x) +
                   GetD(DeviceVars).y * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u + GetDT(DeviceVars) * GetInterimValue(i, j, DeviceVars, DeviceInterim).x) -
                   GetD(DeviceVars).x * (GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).v + GetDT(DeviceVars) * GetInterimValue(i, j + 1, DeviceVars, DeviceInterim).y) +
                   GetD(DeviceVars).x * GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v;
            SetLinearValue(i, j, var, 0, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
            var = (2 * GetDT(DeviceVars) * (GetD(DeviceVars).y / GetD(DeviceVars).x)) + (GetDT(DeviceVars) * (GetD(DeviceVars).x / GetD(DeviceVars).y));
            SetLinearValue(i, j, var, 1, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
            SetLinearValue(i, j, 0.0, 4, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        }
    }
}

__device__ void PhysicsCUDA::BuildRightSide(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ) {
    double var;
    if (j == GetSPLITS(DeviceVars).y - 1) {
        if (i > 0 && i < GetSPLITS(DeviceVars).x - 1) {
            var = -GetD(DeviceVars).y * (GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).u + GetDT(DeviceVars) * GetInterimValue(i + 1, j, DeviceVars, DeviceInterim).x) +
                   GetD(DeviceVars).y * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u + GetDT(DeviceVars) * GetInterimValue(i, j, DeviceVars, DeviceInterim).x) -
                   GetD(DeviceVars).x * GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).v +
                   GetD(DeviceVars).x * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v + GetDT(DeviceVars) * GetInterimValue(i, j, DeviceVars, DeviceInterim).y);
            SetLinearValue(i, j, var, 0, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
            var = (2 * GetDT(DeviceVars) * (GetD(DeviceVars).y / GetD(DeviceVars).x)) + (GetDT(DeviceVars) * (GetD(DeviceVars).x / GetD(DeviceVars).y));
            SetLinearValue(i, j, var, 1, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
            SetLinearValue(i, j, 0.0, 5, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        }
    }
}

__device__ void PhysicsCUDA::BuildTopSide(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ) {
    double var;
    if (i == 0) {
        if (j > 0 && j < GetSPLITS(DeviceVars).y - 1) {
            var = -GetD(DeviceVars).y * (GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).u + GetDT(DeviceVars) * GetInterimValue(i + 1, j, DeviceVars, DeviceInterim).x) +
                   GetD(DeviceVars).y * GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u -
                   GetD(DeviceVars).x * (GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).v + GetDT(DeviceVars) * GetInterimValue(i, j + 1, DeviceVars, DeviceInterim).y) +
                   GetD(DeviceVars).x * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v + GetDT(DeviceVars) * GetInterimValue(i, j, DeviceVars, DeviceInterim).y);
            SetLinearValue(i, j, var, 0, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
            var = (GetDT(DeviceVars) * (GetD(DeviceVars).y / GetD(DeviceVars).x)) + (2 * GetDT(DeviceVars) * (GetD(DeviceVars).x / GetD(DeviceVars).y));
            SetLinearValue(i, j, var, 1, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
            SetLinearValue(i, j, 0.0, 3, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        }
    }
}

__device__ void PhysicsCUDA::BuildBottomSide(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ) {
    double var;
    if (i == GetSPLITS(DeviceVars).x - 1) {
        if (j > 0 && j < GetSPLITS(DeviceVars).y - 1) {
            var = -GetD(DeviceVars).y * GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).u +
                   GetD(DeviceVars).y * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u + GetDT(DeviceVars) * GetInterimValue(i, j, DeviceVars, DeviceInterim).x) -
                   GetD(DeviceVars).x * (GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).v + GetDT(DeviceVars) * GetInterimValue(i, j + 1, DeviceVars, DeviceInterim).y) +
                   GetD(DeviceVars).x * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v + GetDT(DeviceVars) * GetInterimValue(i, j, DeviceVars, DeviceInterim).y);
            SetLinearValue(i, j, var, 0, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
            var = (GetDT(DeviceVars) * (GetD(DeviceVars).y / GetD(DeviceVars).x)) + (2 * GetDT(DeviceVars) * (GetD(DeviceVars).x / GetD(DeviceVars).y));
            SetLinearValue(i, j, var, 1, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
            SetLinearValue(i, j, 0.0, 2, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        }
    }
}

__device__ void PhysicsCUDA::BuildInterior(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ) {
    double var;
    if (i > 0 && i < GetSPLITS(DeviceVars).x - 1) {
        if (j > 0 && j < GetSPLITS(DeviceVars).y - 1) {
            var = -GetD(DeviceVars).y * (GetMatrixValue(i + 1, j, DeviceVars, DeviceMatrix).u + GetDT(DeviceVars) * GetInterimValue(i + 1, j, DeviceVars, DeviceInterim).x) +
                   GetD(DeviceVars).y * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u + GetDT(DeviceVars) * GetInterimValue(i, j, DeviceVars, DeviceInterim).x) -
                   GetD(DeviceVars).x * (GetMatrixValue(i, j + 1, DeviceVars, DeviceMatrix).v + GetDT(DeviceVars) * GetInterimValue(i, j + 1, DeviceVars, DeviceInterim).y) +
                   GetD(DeviceVars).x * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v + GetDT(DeviceVars) * GetInterimValue(i, j, DeviceVars, DeviceInterim).y);
            SetLinearValue(i, j, var, 0, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
            var = (2 * GetDT(DeviceVars) * (GetD(DeviceVars).y / GetD(DeviceVars).x)) + (2 * GetDT(DeviceVars) * (GetD(DeviceVars).x / GetD(DeviceVars).y));
            SetLinearValue(i, j, var, 1, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        }
    }
}

__device__ void PhysicsCUDA::BuildLinearSystem(int i, int j, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ) {
    SetBaseAValues(i, j, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
    BuildTopLeft(i, j, DeviceVars, DeviceMatrix, DeviceInterim, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
    BuildTopRight(i, j, DeviceVars, DeviceMatrix, DeviceInterim, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
    BuildBottomLeft(i, j, DeviceVars, DeviceMatrix, DeviceInterim, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
    BuildBottomRight(i, j, DeviceVars, DeviceMatrix, DeviceInterim, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
    BuildLeftSide(i, j, DeviceVars, DeviceMatrix, DeviceInterim, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
    BuildRightSide(i, j, DeviceVars, DeviceMatrix, DeviceInterim, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
    BuildTopSide(i, j, DeviceVars, DeviceMatrix, DeviceInterim, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
    BuildBottomSide(i, j, DeviceVars, DeviceMatrix, DeviceInterim, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
    BuildInterior(i, j, DeviceVars, DeviceMatrix, DeviceInterim, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
    return;
}

__device__ void PhysicsCUDA::FindValidEntries(int i, int j, double* DeviceVars, double* DeviceNzCoeffMat, int* PrefixSum) {
    int index = (j * GetSPLITS(DeviceVars).x * GetSPLITS(DeviceVars).y) + i;
    int nnz = 5 * GetSPLITS(DeviceVars).x * GetSPLITS(DeviceVars).y;
    if (index < nnz) {
        if (!DeviceNzCoeffMat[index]) { PrefixSum[index] = 0; }
        if (DeviceNzCoeffMat[index]) { PrefixSum[index] = 1; }
    }
    //if (i == 0 && j == 0) {
    //    for (int x = 0; x < nnz; x++) {
    //        printf("Index = %i, Value = %f, PrefixSum = %i\n", x, DeviceNzCoeffMat[x], PrefixSum[x]);
    //    }
    //}
}

__device__ void PhysicsCUDA::SumValidEntries(int i, int j, double* DeviceVars, double* DeviceNzCoeffMat, int* PrefixSum) {
    int index = (j * GetSPLITS(DeviceVars).x * GetSPLITS(DeviceVars).y) + i;
    int nnz = 5 * GetSPLITS(DeviceVars).x * GetSPLITS(DeviceVars).y;
    if (index > 0 && index < nnz) {
        int cdf_val = 0;
        for (int x = 0; x < index; x++) { cdf_val += PrefixSum[x]; }
        __syncthreads();
        PrefixSum[index] = cdf_val;
    }
    //if (i == 0 && j == 0) {
    //    for (int x = 0; x < nnz; x++) {
    //        printf("Index = %i, Value = %f, PrefixSum = %i\n", x, DeviceNzCoeffMat[x], PrefixSum[x]);
    //    }
    //}
}


__device__ void PhysicsCUDA::ShiftDeviceLinearSystem(int i, int j, double* DeviceVars, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ, int* PrefixSum) {
    int index = (j * GetSPLITS(DeviceVars).x * GetSPLITS(DeviceVars).y) + i;
    int nnz = 5 * GetSPLITS(DeviceVars).x * GetSPLITS(DeviceVars).y;
    if (index > 0 && index < nnz) {
        if (DeviceNzCoeffMat[index]) {
            DeviceNzCoeffMat[PrefixSum[index]] = DeviceNzCoeffMat[index];
            DeviceSparseIndexI[PrefixSum[index]] = DeviceSparseIndexI[index];
            DeviceSparseIndexJ[PrefixSum[index]] = DeviceSparseIndexJ[index];
        }
        else { ; }
    }
    else if (index == 0) { DeviceNzCoeffMat[0] = DeviceNzCoeffMat[0]; }
}

__device__ void PhysicsCUDA::BuildSparseMatrixForSolution(int i, int j, double* DeviceVars, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ, int* PrefixSum) {
    FindValidEntries(i, j, DeviceVars, DeviceNzCoeffMat, PrefixSum);
    __syncthreads();
    SumValidEntries(i, j, DeviceVars, DeviceNzCoeffMat, PrefixSum);
    __syncthreads();
    ShiftDeviceLinearSystem(i, j, DeviceVars, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ, PrefixSum);
}

__device__ double PhysicsCUDA::ComputeIteration(int i, int j, int dim, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim) {
    if (dim == 0) {
        double var = GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u + (GetDT(DeviceVars) * GetInterimValue(i, j, DeviceVars, DeviceInterim).x) -
                     GetDT(DeviceVars) * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).p - GetMatrixValue(i - 1, j, DeviceVars, DeviceMatrix).p) / GetD(DeviceVars).x;
        return var;
    }
    else if (dim == 1) {
        double var = GetMatrixValue(i, j, DeviceVars, DeviceMatrix).v + (GetDT(DeviceVars) * GetInterimValue(i, j, DeviceVars, DeviceInterim).y) -
                     GetDT(DeviceVars) * (GetMatrixValue(i, j, DeviceVars, DeviceMatrix).p - GetMatrixValue(i, j - 1, DeviceVars, DeviceMatrix).p) / GetD(DeviceVars).y;
        return var;
    }
}

/*void PhysicsCUDA::ThrowCoefficients() {
    std::ofstream CoeFile;
    CoeFile.open("./Output/Coefficients.txt");
    CoeFile << "| B | AC | AIP | AIN | AJP | AJN |" << std::endl;
    for (int i = 0; i < GetSPLITS().x * GetSPLITS().y; i++) {
        if (i < ((GetSPLITS().x * GetSPLITS().y) - GetSPLITS().x)) {
            CoeFile << GetLinearValue(i).Bvec << " , " << GetLinearValue(i).Acen << " , "
                    << GetLinearValue(i).Aipos << " , " << GetLinearValue(i + 1).Aisub << " , "
                    << GetLinearValue(i + (int)GetSPLITS().x).Ajpos << " , " << GetLinearValue(i).Ajsub << std::endl;
        }
        else {
            if (i < ((GetSPLITS().x * GetSPLITS().y) - 1)) {
                CoeFile << GetLinearValue(i).Bvec << " , " << GetLinearValue(i).Acen << " , "
                    << GetLinearValue(i).Aipos << " , " << GetLinearValue(i + 1).Aisub << std::endl;
            }
            else {
                CoeFile << GetLinearValue(i).Bvec << " , " << GetLinearValue(i).Acen << std::endl;
            }
        }
    }
    CoeFile.close();
    return;
}*/