#include "FVMCUDA.cuh"

__global__ void LinearSystemKernel(PhysicsCUDA* CPUinstance) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 1 && i < CPUinstance->GetSPLITS().x) {
        if (j > 0 && j < CPUinstance->GetSPLITS().y) {
            CPUinstance->BuildLinearSystem(i, j);
        }
    }
}

__host__ void PhysicsCUDA::LinearSystemGPUWrapper() {
    unsigned int SIZE = (int)GetSPLITS().x * (int)GetSPLITS().y;
    dim3 SPLIT((int)GetSPLITS().x, (int)GetSPLITS().y);
    LinearSystemKernel <<<SPLIT, SIZE >>> (this);
    std::cout << "Kernel Launched" << std::endl;
    return;
}

//__host__ PhysicsCUDA PhysicsCUDA::GetDeviceObject() { return *DeviceObject; }

//__host__ void PhysicsCUDA::AllocateObjectMemoryToGPU() {
//    cudaMalloc((void**)DeviceObject, OBJECTSIZE);
//}

__host__ void PhysicsCUDA::CudaErrorChecker(cudaError_t func) {
    if (func == cudaSuccess) { std::cout << "CudaSuccess" << std::endl; }
    else { std::cout << stderr << "cudaFailure: " << cudaGetErrorString(func) << std::endl; }
}

__host__ void PhysicsCUDA::AllocateMatrixMemoryToGPU() {
    CudaErrorChecker(cudaMalloc((void**)&DeviceMatrix, VEC3ALLOCSIZE));

}

__host__ void PhysicsCUDA::AllocateMatrixSolutionToGPU() {
    CudaErrorChecker(cudaMalloc((void**)&DeviceMatrixSolution, VEC3ALLOCSIZE));
}

__host__ void PhysicsCUDA::AllocateInterimMemoryToGPU() {
    CudaErrorChecker(cudaMalloc((void**)&DeviceInterim, VEC2ALLOCSIZE));
}
__host__ void PhysicsCUDA::AllocateInterimSolutionToGPU() {
    CudaErrorChecker(cudaMalloc((void**)&DeviceInterimSolution, VEC2ALLOCSIZE));
}

__host__ void PhysicsCUDA::AllocateLinearSystemMemoryToGPU() {
    CudaErrorChecker(cudaMalloc((void**)&DeviceLinearSystem, DIM6ALLOCSIZE));
}

//__host__ void PhysicsCUDA::SetObjectToGPU() {
//    cudaMemcpy(DeviceObject, this, OBJECTSIZE, cudaMemcpyHostToDevice);
//}

__host__ void PhysicsCUDA::SetMatrixToGPU() {
    std::vector<std::vector<vec3>> SystemMatrix = GetSystemMatrix();
    CudaErrorChecker(cudaMemcpy(DeviceMatrix, &SystemMatrix, SYSTEMSIZE * sizeof(vec3), cudaMemcpyHostToDevice));
}

__host__ void PhysicsCUDA::SetInterimToGPU() {
    std::vector<std::vector<vec2>> InterimMatrix = GetInterimMatrix();
    CudaErrorChecker(cudaMemcpy(DeviceInterim, &InterimMatrix, SYSTEMSIZE * sizeof(vec2), cudaMemcpyHostToDevice));
}

__host__ void PhysicsCUDA::SetLinearSystemToGPU() {
    CudaErrorChecker(cudaMemcpy(DeviceLinearSystem, &LinearSystemMatrix, SYSTEMSIZE * sizeof(dim6), cudaMemcpyHostToDevice));
}

__host__ void PhysicsCUDA::GetMatrixFromGPU() {
    std::vector<std::vector<vec3>> SystemMatrix = GetSystemMatrix();
    CudaErrorChecker(cudaMemcpy(&SystemMatrix, DeviceMatrixSolution, SYSTEMSIZE * sizeof(vec3), cudaMemcpyDeviceToHost));
}

__host__ void PhysicsCUDA::GetInterimFromGPU() {
    std::vector<std::vector<vec2>> InterimMatrix = GetInterimMatrix();
    CudaErrorChecker(cudaMemcpy(&InterimMatrix , DeviceInterimSolution, SYSTEMSIZE * sizeof(vec2), cudaMemcpyDeviceToHost));
}

__host__ void PhysicsCUDA::GetLinearSystemFromGPU() {
    CudaErrorChecker(cudaMemcpy(&LinearSystemMatrix, DeviceLinearSystem, SYSTEMSIZE * sizeof(dim6), cudaMemcpyDeviceToHost));
}

//__host__ void PhysicsCUDA::DeviceObjectCleanUp() {
//    cudaFree(DeviceObject);
//}

__host__ void PhysicsCUDA::DeviceMatrixCleanUp() {
	cudaFree(DeviceMatrix);
	cudaFree(DeviceMatrixSolution);
}

__host__ void PhysicsCUDA::DeviceInterimCleanUp() {
	cudaFree(DeviceInterim);
	cudaFree(DeviceInterimSolution);
}

__host__ void PhysicsCUDA::DeviceLinearSystemCleanUp() {
    cudaFree(DeviceLinearSystem);
}

__device__ vec3 PhysicsCUDA::GetMatrixValue(int i, int j) {
    return DeviceMatrix[i][j];
}

__device__ vec2 PhysicsCUDA::GetInterimValue(int i, int j) {
    return DeviceInterim[i][j];
}

__device__ void PhysicsCUDA::SetMatrixValue(int i, int j, double var, const char* dim) {
    if (dim == "u") { DeviceMatrix[i][j].u = var; }
    else if (dim == "v") { DeviceMatrix[i][j].v = var; }
    else if (dim == "p") { DeviceMatrix[i][j].p = var; } 
//    else { std::cout << "No dimension found" << std::endl; }
}

__device__ void PhysicsCUDA::SetInterimValue(int i, int j, double var, const char* dim) {
    if (dim == "u") { DeviceInterimSolution[i][j].x = var; }
    else if (dim == "v") { DeviceInterimSolution[i][j].y = var; }
//    else { std::cout << "No dimension found" << std::endl; }
}

__device__ vec6 PhysicsCUDA::InterpolateVelocities(int i, int j, const char* dim) {
    if (dim == "x") {
        double UEAST = 0.5 * (GetMatrixValue(i + 1, j).u + GetMatrixValue(i, j).u);
        double UWEST = 0.5 * (GetMatrixValue(i, j).u + GetMatrixValue(i - 1, j).u);
        double VNORTH = 0.5 * (GetMatrixValue(i - 1, j + 1).v + GetMatrixValue(i, j + 1).v);
        double VSOUTH = 0.5 * (GetMatrixValue(i - 1, j).v + GetMatrixValue(i, j).v);

        if (j == 0) {
            double UNORTH = 0.5 * (GetMatrixValue(i, j + 1).u + GetMatrixValue(i, j).u);
            double USOUTH = GetVelocityBoundary().E;
            return vec6(UEAST, UWEST, UNORTH, USOUTH, VNORTH, VSOUTH);
        }
        else if (j == GetSPLITS().y - 1) {
            double UNORTH = GetVelocityBoundary().W;
            double USOUTH = 0.5 * (GetMatrixValue(i, j).u + GetMatrixValue(i, j - 1).u);
            return vec6(UEAST, UWEST, UNORTH, USOUTH, VNORTH, VSOUTH);
        }
        else {
            double UNORTH = 0.5 * (GetMatrixValue(i, j + 1).u + GetMatrixValue(i, j).u);
            double USOUTH = 0.5 * (GetMatrixValue(i, j).u + GetMatrixValue(i, j - 1).u);
            return vec6(UEAST, UWEST, UNORTH, USOUTH, VNORTH, VSOUTH);
        }
    }
    else if (dim == "y") {
        double VNORTH = 0.5 * (GetMatrixValue(i, j + 1).v + GetMatrixValue(i, j).v);
        double VSOUTH = 0.5 * (GetMatrixValue(i, j).v + GetMatrixValue(i, j - 1).v);
        double UEAST = 0.5 * (GetMatrixValue(i + 1, j - 1).u + GetMatrixValue(i + 1, j).u);
        double UWEST = 0.5 * (GetMatrixValue(i, j - 1).u + GetMatrixValue(i, j).u);

        if (i == 0) {
            double VEAST = 0.5 * (GetMatrixValue(i + 1, j).v + GetMatrixValue(i, j).v);
            double VWEST = GetVelocityBoundary().S;
            return vec6(VEAST, VWEST, VNORTH, VSOUTH, UEAST, UWEST);
        }
        else if (i == GetSPLITS().x - 1) {
            double VEAST = GetVelocityBoundary().N;
            double VWEST = 0.5 * (GetMatrixValue(i, j).v + GetMatrixValue(i - 1, j).v);
            return vec6(VEAST, VWEST, VNORTH, VSOUTH, UEAST, UWEST);
        }
        else {
            double VEAST = 0.5 * (GetMatrixValue(i + 1, j).v + GetMatrixValue(i, j).v);
            double VWEST = 0.5 * (GetMatrixValue(i, j).v + GetMatrixValue(i - 1, j).v);
            return vec6(VEAST, VWEST, VNORTH, VSOUTH, UEAST, UWEST);
        }
    }
//    else {
//        std::cout << "No dimension found" << std::endl;
//        throw - 1;
//    }
}

__device__ double PhysicsCUDA::ComputeAdvection(int i, int j, const char* dim) {
    if (dim == "x") {
        vec6 var1 = InterpolateVelocities(i, j, dim);
        double XX = (var1.E * var1.E - var1.W * var1.W) / GetD().x;
        double XY = (var1.N * var1.EN - var1.S * var1.WS) / GetD().y;
        return -(XX + XY);
    }

    else if (dim == "y") {
        vec6 var2 = InterpolateVelocities(i, j, dim);
        double YY = (var2.N * var2.N - var2.S * var2.S) / GetD().y;
        double YX = (var2.E * var2.EN - var2.W * var2.WS) / GetD().x;
        return -(YY + YX);
    }
//    else {
//        std::cout << "No dimension found" << std::endl;
//        throw - 1;
//    }
}

__device__ double PhysicsCUDA::ComputeDiffusion(int i, int j, const char* dim) {
    if (dim == "x") {
        double XDXe = -2 * GetNU() * (GetMatrixValue(i + 1, j).u - GetMatrixValue(i, j).u) / GetD().x;
        double XDXw = -2 * GetNU() * (GetMatrixValue(i, j).u - GetMatrixValue(i - 1, j).u) / GetD().x;

        if (j == GetSPLITS().y - 1) {
            double XDYn = -GetNU() * (GetVelocityBoundary().E - GetMatrixValue(i, j).u) / (GetD().y / 2) -
                GetNU() * (GetMatrixValue(i, j + 1).v - GetMatrixValue(i - 1, j + 1).v) / GetD().x;
            double XDYs = -GetNU() * (GetMatrixValue(i, j).u - GetMatrixValue(i, j - 1).u) / GetD().y -
                GetNU() * (GetMatrixValue(i, j).v - GetMatrixValue(i - 1, j).v) / GetD().x;
            return (XDXe - XDXw) / GetD().x + (XDYn - XDYs) / GetD().y;
        }
        if (j == 0) {
            double XDYn = -GetNU() * (GetMatrixValue(i, j + 1).u - GetMatrixValue(i, j).u) / GetD().y -
                GetNU() * (GetMatrixValue(i, j + 1).v - GetMatrixValue(i - 1, j + 1).v) / GetD().x;
            double XDYs = -GetNU() * (GetMatrixValue(i, j).u - GetVelocityBoundary().W) / (GetD().y / 2) -
                GetNU() * (GetMatrixValue(i, j).v - GetMatrixValue(i - 1, j).v) / GetD().x;
            return (XDXe - XDXw) / GetD().x + (XDYn - XDYs) / GetD().y;
        }
        else {
            double XDYn = -GetNU() * (GetMatrixValue(i, j + 1).u - GetMatrixValue(i, j).u) / GetD().y -
                GetNU() * (GetMatrixValue(i, j + 1).v - GetMatrixValue(i - 1, j + 1).v) / GetD().x;
            double XDYs = -GetNU() * (GetMatrixValue(i, j).u - GetMatrixValue(i, j - 1).u) / GetD().y -
                GetNU() * (GetMatrixValue(i, j).v - GetMatrixValue(i - 1, j).v) / GetD().x;
            return (XDXe - XDXw) / GetD().x + (XDYn - XDYs) / GetD().y;
        }
    }
    else if (dim == "y") {
        double YDYn = -2 * GetNU() * (GetMatrixValue(i, j + 1).v - GetMatrixValue(i, j).v) / GetD().y;
        double YDYs = -2 * GetNU() * (GetMatrixValue(i, j).v - GetMatrixValue(i, j - 1).v) / GetD().y;

        if (i == GetSPLITS().x - 1) {
            double YDXe = -GetNU() * (GetVelocityBoundary().S - GetMatrixValue(i, j).v) / (GetD().x / 2) -
                GetNU() * (GetMatrixValue(i + 1, j).u - GetMatrixValue(i + 1, j - 1).u) / GetD().y;
            double YDXw = -GetNU() * (GetMatrixValue(i, j).v - GetMatrixValue(i - 1, j).v) / GetD().x -
                GetNU() * (GetMatrixValue(i, j).u - GetMatrixValue(i, j - 1).u) / GetD().y;
            return (YDYn - YDYs) / GetD().y + (YDXe - YDXw) / GetD().x;
        }
        if (i == 0) {
            double YDXe = -GetNU() * (GetMatrixValue(i + 1, j).v - GetMatrixValue(i, j).v) / GetD().x -
                GetNU() * (GetMatrixValue(i + 1, j).u - GetMatrixValue(i + 1, j - 1).u) / GetD().y;
            double YDXw = -GetNU() * (GetMatrixValue(i, j).v - GetVelocityBoundary().N) / (GetD().x / 2) -
                GetNU() * (GetMatrixValue(i, j).u - GetMatrixValue(i, j - 1).u) / GetD().y;
            return (YDYn - YDYs) / GetD().y + (YDXe - YDXw) / GetD().x;
        }
        else {
            double YDXe = -GetNU() * (GetMatrixValue(i + 1, j).v - GetMatrixValue(i, j).v) / GetD().x -
                GetNU() * (GetMatrixValue(i + 1, j).u - GetMatrixValue(i + 1, j - 1).u) / GetD().y;
            double YDXw = -GetNU() * (GetMatrixValue(i, j).v - GetMatrixValue(i - 1, j).v) / GetD().x -
                GetNU() * (GetMatrixValue(i, j).u - GetMatrixValue(i, j - 1).u) / GetD().y;
            return (YDYn - YDYs) / GetD().y + (YDXe - YDXw) / GetD().x;
        }
    }
//    else {
//        std::cout << "No dimension found" << std::endl;
//        throw - 1;
//    }
}

__device__ void PhysicsCUDA::ComputeMomentum(int i, int j, const char* dim) {
    double var = (ComputeAdvection(i, j, dim) - ComputeDiffusion(i, j, dim));
    if (dim == "y") {
        //std::cout << "I , J = " << i << " , " << j << std::endl;
        //std::cout << "ADV = " << ComputeAdvection(i, j, dim) << std::endl;
        //std::cout << "DIF = " << ComputeDiffusion(i, j, dim) << std::endl << std::endl;
    }
    if (dim == "x") { dim = "u"; SetInterimValue(i, j, var, dim); }
    else if (dim == "y") { dim = "v"; SetInterimValue(i, j, var, dim); }
}

__host__ void PhysicsCUDA::BuildCoeffMat() {
    LinearSystemMatrix = std::vector<std::vector<dim6>>((int)GetSPLITS().x, std::vector<dim6>((int)GetSPLITS().y));
    LinearSystemRESHAPED = std::vector<dim6>((int)GetSPLITS().x * (int)GetSPLITS().y);
}

__device__ void PhysicsCUDA::SetBaseAValues(int i, int j) {
    int POINTS = (int)GetSPLITS().x * (int)GetSPLITS().y;
    if (i < GetSPLITS().x && j < GetSPLITS().y) {
        DeviceLinearSystem[i][j].Aipos = -GetDT() * (GetD().y / GetD().x); // Havent set this to GPU
        DeviceLinearSystem[i][j].Aisub = -GetDT() * (GetD().y / GetD().x);
        DeviceLinearSystem[i][j].Ajpos = -GetDT() * (GetD().x / GetD().y);
        DeviceLinearSystem[i][j].Ajsub = -GetDT() * (GetD().x / GetD().y);
    }
}

__device__ void PhysicsCUDA::BuildTopLeft(int i, int j) {
    if (i == 0 && j == 0) {
        DeviceLinearSystem[i][j].Bvec = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
            GetD().y * GetMatrixValue(i, j).u -
            GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
            GetD().x * GetMatrixValue(i, j).v;
        DeviceLinearSystem[i][j].Acen = (GetDT() * (GetD().y / GetD().x)) + (GetDT() * (GetD().x / GetD().y));
        DeviceLinearSystem[i][j].Aisub = 0;
        DeviceLinearSystem[i][j].Ajpos = 0;
    }
}

__device__ void PhysicsCUDA::BuildTopRight(int i, int j) { // Bottom Left
    if (i == 0 && j == (int)GetSPLITS().y - 1) {
        DeviceLinearSystem[i][j].Bvec = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
            GetD().y * GetMatrixValue(i, j).u -
            GetD().x * GetMatrixValue(i, j + 1).v +
            GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y);
        DeviceLinearSystem[i][j].Acen = (GetDT() * (GetD().y / GetD().x)) + GetDT() * (GetD().x / GetD().y);
        DeviceLinearSystem[i][j].Ajsub = 0;
        DeviceLinearSystem[i][j].Aisub = 0;
    }
}

__device__ void PhysicsCUDA::BuildBottomLeft(int i, int j) { // Top Right
    if (i == (int)GetSPLITS().x - 1 && j == 0) {
        DeviceLinearSystem[i][j].Bvec = -GetD().y * GetMatrixValue(i + 1, j).u +
            GetD().y * (GetMatrixValue(i, j).u + GetDT() * GetInterimValue(i, j).x) -
            GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
            GetD().x * GetMatrixValue(i, j).v;
        DeviceLinearSystem[i][j].Acen = (GetDT() * (GetD().y / GetD().x)) + (GetDT() * (GetD().x / GetD().y));
        DeviceLinearSystem[i][j].Aipos = 0;
        DeviceLinearSystem[i][j].Ajpos = 0;
    }
}

__device__ void PhysicsCUDA::BuildBottomRight(int i, int j) {
    if (i == (int)GetSPLITS().x - 1 && j == (int)GetSPLITS().y - 1) {
        DeviceLinearSystem[i][j].Bvec = -GetD().y * GetMatrixValue(i + 1, j).u +
            GetD().y * (GetMatrixValue(j, j).u + GetDT() * GetInterimValue(i, j).x) -
            GetD().x * GetMatrixValue(i, j + 1).v +
            GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y);
        DeviceLinearSystem[i][j].Acen = (GetDT() * (GetD().y / GetD().x)) + (GetDT() * (GetD().x / GetD().y));
        DeviceLinearSystem[i][j].Ajsub = 0;
        DeviceLinearSystem[i][j].Aipos = 0;
    }
}

__device__ void PhysicsCUDA::BuildLeftSide(int i, int j) { // Top side
    if (j == 0) {
        if ( i == 1 && i < GetSPLITS().x - 1) {
            DeviceLinearSystem[i][j].Bvec = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
                GetD().y * (GetMatrixValue(i, j).u + GetDT() * GetInterimValue(i, j).x) -
                GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
                GetD().x * GetMatrixValue(i, j).v;
            DeviceLinearSystem[i][j].Acen = (2 * GetDT() * (GetD().y / GetD().x)) + (GetDT() * (GetD().x / GetD().y));
            DeviceLinearSystem[i][j].Ajpos = 0;
        }
    }
}

__device__ void PhysicsCUDA::BuildRightSide(int i, int j) {
    if (j == GetSPLITS().y - 1) {
        if (i > 0 && i < GetSPLITS().x - 1) {
            DeviceLinearSystem[i][j].Bvec = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
                GetD().y * (GetMatrixValue(i, j).u + GetDT() * GetInterimValue(i, j).x) -
                GetD().x * GetMatrixValue(i, j + 1).v +
                GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y);
            DeviceLinearSystem[i][j].Acen = (2 * GetDT() * (GetD().y / GetD().x)) + (GetDT() * (GetD().x / GetD().y));
            DeviceLinearSystem[i][j].Ajsub = 0;
        }
    }
}

__device__ void PhysicsCUDA::BuildTopSide(int i, int j) {
    if (i == 0) {
        if (j > 0 && j < GetSPLITS().y - 1) {
            DeviceLinearSystem[i][j].Bvec = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
                GetD().y * GetMatrixValue(i, j).u -
                GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
                GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y);
            DeviceLinearSystem[i][j].Acen = (GetDT() * (GetD().y / GetD().x)) + (2 * GetDT() * (GetD().x / GetD().y));
            DeviceLinearSystem[i][j].Aisub = 0;
        }
    }
}

__device__ void PhysicsCUDA::BuildBottomSide(int i, int j) {
    if (i == GetSPLITS().x - 1) {
        if (j > 0 && j < GetSPLITS().y - 1) {
            DeviceLinearSystem[i][j].Bvec = -GetD().y * GetMatrixValue(i + 1, j).u +
                GetD().y * (GetMatrixValue(i, j).u + GetDT() * GetInterimValue(i, j).x) -
                GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
                GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y);
            DeviceLinearSystem[i][j].Acen = (GetDT() * (GetD().y / GetD().x)) + (2 * GetDT() * (GetD().x / GetD().y));
            DeviceLinearSystem[i][j].Aipos = 0;
        }
    }
}

__device__ void PhysicsCUDA::BuildInterior(int i, int j) {
    if (i > 0 && i < GetSPLITS().x - 1) {
        if (j > 0 && j < GetSPLITS().y - 1) {
            DeviceLinearSystem[i][j].Bvec = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
                GetD().y * (GetMatrixValue(i, j).u + GetDT() * GetInterimValue(i, j).x) -
                GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
                GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y);
            DeviceLinearSystem[i][j].Acen = (2 * GetDT() * (GetD().y / GetD().x)) + (2 * GetDT() * (GetD().x / GetD().y));
        }
    }
}

__host__ void PhysicsCUDA::ReshapeCoefficients() {
    int POINTS = (int)GetSPLITS().x * (int)GetSPLITS().y;
    for (int j = 0; j < GetSPLITS().x; j++) {
        for (int i = 0; i < GetSPLITS().y; i++) {
            //std::cout << LinearSystemMatrix[i][j].Bvec << " , ";
            LinearSystemRESHAPED[(j * (int)GetSPLITS().x) + i] = LinearSystemMatrix[i][j];
        }
        //std::cout << std::endl;
    }
}

__device__ void PhysicsCUDA::BuildLinearSystem(int i, int j) {
    SetBaseAValues(i, j);
    BuildTopLeft(i, j);
    BuildTopRight(i, j);
    BuildBottomLeft(i, j);
    BuildBottomRight(i, j);
    BuildLeftSide(i, j);
    BuildRightSide(i, j);
    BuildTopSide(i, j);
    BuildBottomSide(i, j);
    BuildInterior(i, j);

    //Acoe = nc::diag<double>(Ajsub, -XSPLIT) + // <-- DON'T DELETE THIS IS IMPORTANT
    //    nc::diag<double>(Aisub, -1) +
    //    nc::diag<double>(Acen, 0) +
    //    nc::diag<double>(Aipos, 1) +
     //   nc::diag<double>(Ajpos, XSPLIT);
    return;
}

__device__ void PhysicsCUDA::ComputeIteration(int i, int j, const char* dim) {
    if (dim == "x") {
        double var = GetMatrixValue(i, j).u + (GetDT() * GetInterimValue(i, j).x) - GetDT() * (GetMatrixValue(i, j).p - GetMatrixValue(i - 1, j).p) / GetD().x;
        dim = "u";
        SetMatrixValue(i, j, var, dim);
    }
    else if (dim == "y") {
        double var = GetMatrixValue(i, j).v + (GetDT() * GetInterimValue(i, j).y) - GetDT() * (GetMatrixValue(i, j).p - GetMatrixValue(i, j - 1).p) / GetD().y;
        dim = "v";
        SetMatrixValue(i, j, var, dim);
    }
}

__host__ void PhysicsCUDA::ThrowCoefficients() {
    std::ofstream CoeFile;
    CoeFile.open("./Output/Coefficients.txt");
    CoeFile << "| B | AC | AIP | AIN | AJP | AJN |" << std::endl;
    int it = 0;
    for (int i = 0; i < GetSPLITS().x * GetSPLITS().y; i++) {
        if (it < ((GetSPLITS().x * GetSPLITS().y) - GetSPLITS().x)) {
            CoeFile << LinearSystemRESHAPED[i].Bvec << " , " << LinearSystemRESHAPED[i].Acen << " , "
                << LinearSystemRESHAPED[i].Aipos << " , " << LinearSystemRESHAPED[i + 1].Aisub << " , "
                << LinearSystemRESHAPED[i + (int)GetSPLITS().x].Ajpos << " , " << LinearSystemRESHAPED[i].Ajsub << std::endl;
        }
        else {
            if (it < ((GetSPLITS().x * GetSPLITS().y) - 1)) {
                CoeFile << LinearSystemRESHAPED[i].Bvec << " , " << LinearSystemRESHAPED[i].Acen << " , "
                    << LinearSystemRESHAPED[i].Aipos << " , " << LinearSystemRESHAPED[i + 1].Aisub << std::endl;
            }
            else {
                CoeFile << LinearSystemRESHAPED[i].Bvec << " , " << LinearSystemRESHAPED[i].Acen << std::endl;
            }
        }
        it++;
    }
    CoeFile.close();
    return;
}