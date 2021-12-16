#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
#include <python.h>
#endif

#include "IncompressCUDA.cuh"

__global__ void InterimMomentumKernel(IncompressCUDA* CPUinstance, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    double var;
    //if (j == 0 && i == 0) { printf("%f \n", CPUinstance->GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u); }
    //if (j == 0 && i == 0) { CPUinstance->SetMatrixValue(i, j, 0.0, dim, DeviceVars, DeviceMatrix); }
    //if (j == 0 && i == 0) { printf("%f \n", CPUinstance->GetMatrixValue(i, j, DeviceVars, DeviceMatrix).u); }
    if (i > 0 && i < CPUinstance->GetSPLITS(DeviceVars).x) {
        if (j < CPUinstance->GetSPLITS(DeviceVars).y) {
            int index = (j * CPUinstance->GetSPLITS(DeviceVars).y + i);
            var = CPUinstance->ComputeMomentum(i, j, 0, DeviceVars, DeviceMatrix);
            CPUinstance->SetInterimValue(i, j, var, 0, DeviceVars, DeviceInterim);
            //printf("%f", CPUinstance->GetSPLITS(DeviceVars).x);
            //printf("%i %i %i \n", i, j, index);
            //printf("\n");
            //printf("%i", j);
            //printf("\n");
            //printf("%i", index);
            //printf("I = %i, J = %i, var =  %f \n", i, j, CPUinstance->ComputeDiffusion(i, j, 0, DeviceVars, DeviceMatrix));
        }
    }
    if (i >= 0 && i < CPUinstance->GetSPLITS(DeviceVars).x + 1) {
        if (j >= 1 && j < CPUinstance->GetSPLITS(DeviceVars).y + 1) {
            var = CPUinstance->ComputeMomentum(i, j, 1, DeviceVars, DeviceMatrix);
            CPUinstance->SetInterimValue(i, j, var, 1, DeviceVars, DeviceInterim);
        }
    }
}

__global__ void LinearSystemKernel(PhysicsCUDA* CPUinstance, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    //if (i == 0 && j == 0) { printf("I = %i, J = %i, SPLITSx = %f \n", i, j, CPUinstance->GetSPLITS(DeviceVars).x); }
    if (i < CPUinstance->GetSPLITS(DeviceVars).x) {
        if (j < CPUinstance->GetSPLITS(DeviceVars).y) {
            CPUinstance->BuildLinearSystem(i, j, DeviceVars, DeviceMatrix, DeviceInterim, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        }
    }
    //if (i < CPUinstance->GetSPLITS(DeviceVars).x * CPUinstance->GetSPLITS(DeviceVars).x) {
    //    if (j < CPUinstance->GetSPLITS(DeviceVars).y * CPUinstance->GetSPLITS(DeviceVars).y) {
    //        CPUinstance->BuildSparseMatrixForSolution(i, j, DeviceVars, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
    //    }
    //}
}

__global__ void TrueMomentumKernel(IncompressCUDA* CPUinstance, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    double var;
    if (i >= 1 && i < CPUinstance->GetSPLITS(DeviceVars).x) {
        if (j < CPUinstance->GetSPLITS(DeviceVars).y) {
            var = CPUinstance->ComputeIteration(i, j, 0, DeviceVars, DeviceMatrix, DeviceInterim);
            CPUinstance->SetMatrixValue(i, j, var, 0, DeviceVars, DeviceMatrix);
        }
    }
    if (i < CPUinstance->GetSPLITS(DeviceVars).x) {
        if (j >= 1 && j < CPUinstance->GetSPLITS(DeviceVars).y) {
            var = CPUinstance->ComputeIteration(i, j, 1, DeviceVars, DeviceMatrix, DeviceInterim);
            CPUinstance->SetMatrixValue(i, j, var, 1, DeviceVars, DeviceMatrix);
        }
    }
}

__host__ void IncompressCUDA::InterimMomentumStep() {
    InterimMomentumKernel<<<dim3(64, 64, 4), dim3(32, 32)>>> (this, DeviceVars, DeviceMatrix, DeviceInterim);
    return;
}

__host__ void IncompressCUDA::LinearSystemBuilder () {
    LinearSystemKernel <<<dim3(64, 64, 4), dim3(32, 32) >> > (this, DeviceVars, DeviceMatrix, DeviceInterim, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
    return;
}

__host__ void IncompressCUDA::TrueMomentumStep() {
    TrueMomentumKernel <<<dim3(64, 64, 4), dim3(32, 32) >> > (this, DeviceVars, DeviceMatrix, DeviceInterim);
    //std::cout << "Kernel Opened";
}

__host__ void IncompressCUDA::SystemDriver() {
    AllocateSystemMatrixMemoryToCPU();
    AllocateInterimMatrixMemoryToCPU();
    AllocateSparseIndexMemoryToCPU();
    AllocateColumnIndexMemoryToCPU();
    AllocateCompressedRowMemoryToCPU();
    AllocateLinearSolutionMemoryToCPU();

    //std::cout << "AllocateVariablesMemoryToGPU ->";
    AllocateVariablesMemoryToGPU();
    //std::cout << "AllocateMatrixMemoryToGPU ->";
    AllocateMatrixMemoryToGPU();
    //std::cout << "AllocateInterimMemoryToGPU ->";
    AllocateInterimMemoryToGPU();
    //std::cout << "AllocateLinearSystemMemoryToGPU ->";
    AllocateLinearSystemMemoryToGPU();

    //Py_Initialize();

    //std::cout << "SetVariablesToGPU ->";
    SetVariablesToGPU();

    std::chrono::duration<double> LOOPTIME, EXECUTETIME, ETA;
    auto init = std::chrono::high_resolution_clock::now();
    for (GetCURRENTSTEP(); GetCURRENTSTEP() < GetSIMSTEPS(); IncreaseSTEP()) {
        auto loopTimer = std::chrono::high_resolution_clock::now();

        if (GetCURRENTSTEP() > 0) {
            //if (CheckConvergedExit()) { break; }
            if (CheckDivergedExit()) { break; }
            else { std::cout << "\033[A\33[2K\r"; }
        }
        else { SetKineticEnergy(); }

        ETA = (LOOPTIME * (((double)GetSIMSTEPS() - 1.000) - (double)GetCURRENTSTEP()));
        auto MINUTES = std::chrono::duration_cast<std::chrono::minutes>(ETA);
        auto HOURS = std::chrono::duration_cast<std::chrono::hours>(MINUTES);
        ETA -= MINUTES;
        MINUTES -= HOURS;

        std::cout << GetCURRENTSTEP() << " / " << GetSIMSTEPS() - 1 << " | Estimated time remaining: ";
        std::cout << HOURS.count() << " Hours ";
        std::cout << MINUTES.count() << " Minutes ";
        std::cout << ETA.count() << " Seconds |" << std::endl;

        //std::cout << "SetMatrixToGPU ->";
        SetMatrixToGPU();
        //std::cout << "SetInterimToGPU ->";
        SetInterimToGPU();

        //std::cout << "InterimMomentumStep ->";
        InterimMomentumStep();
        CudaErrorChecker(cudaDeviceSynchronize());

        //std::cout << "SetLinearSystemToGPU ->";
        SetLinearSystemToGPU();
        //std::cout << "LinearSystemBuilder ->";
        LinearSystemBuilder();
        CudaErrorChecker(cudaDeviceSynchronize());
        //std::cout << "GetLinearSystemFromGPU ->";
        GetLinearSystemFromGPU();

        BuildSparseMatrixForSolution();
        FindSparseLinearSolution();
        UpdatePressureValues();
        
        //ThrowSystemVariables();

        //if (GetCURRENTSTEP() == 0) { break; }


        //FILE* fd = _Py_fopen("../x64/Debug/SparseSolver.py", "rb");
        //PyRun_SimpleFileEx(fd, "SparseSolver.py", 1);

        //CatchSolution();
        //TrueMomentumStep();
        //ThrowSystemVariables();

        //ThrowCoefficients();
        //ThrowSystemVariables();

        //FILE* fd = _Py_fopen("../x64/Debug/SparseSolver.py", "rb");
        //PyRun_SimpleFileEx(fd, "SparseSolver.py", 1);

        //if (GetCURRENTSTEP() == 0) { break; }

        //CatchSolution();

        //std::cout << "SetMatrixToGPU ->";
        SetMatrixToGPU();
        //std::cout << "TrueMomentumStep ->";
        TrueMomentumStep();
        CudaErrorChecker(cudaDeviceSynchronize());
        //std::cout << "GetMatrixFromGPU ->";
        GetMatrixFromGPU();

        //ThrowSystemVariables();
        //if (GetCURRENTSTEP() == 0) { break; }

        auto loopEnd = std::chrono::high_resolution_clock::now();
        LOOPTIME = loopEnd - loopTimer;
    }
    if (!CheckConvergedExit()) { LoopBreakOutput(); }
    if (!CheckDivergedExit()) { LoopBreakOutput(); }

    auto end = std::chrono::high_resolution_clock::now();
    EXECUTETIME = end - init;
    std::cout << "System Elapsed Time: " << GetCURRENTSTEP() * GetDT() << " Seconds" << std::endl;
    std::cout << "Loop Execution Time: " << EXECUTETIME.count() << " Seconds" << std::endl;
    
    ThrowSystemVariables();
    //Py_Finalize();

    DeviceVariablesCleanUp();
    DeviceMatrixCleanUp();
    DeviceInterimCleanUp();
    DeviceLinearSystemCleanUp();

    DeAllocateSystemMatrixMemoryOnCPU();
    DeAllocateInterimMatrixMemoryOnCPU();
    DeAllocateSparseIndexMemoryToCPU();
    DeAllocateColumnIndexMemoryToCPU();
    DeAllocateCompressedRowMemoryToCPU();
    DeAllocateLinearSolutionMemoryToCPU();
}