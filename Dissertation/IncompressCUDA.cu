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

__global__ void LinearSystemKernel(PhysicsCUDA* CPUinstance, double* DeviceVars, vec3* DeviceMatrix, vec2* DeviceInterim, double* DeviceRHSVector, double* DeviceNzCoeffMat, int* DeviceSparseIndexI, int* DeviceSparseIndexJ, int* PrefixSum) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    //if (i == 0 && j == 0) { printf("I = %i, J = %i, SPLITSx = %f \n", i, j, CPUinstance->GetSPLITS(DeviceVars).x); }
    if (i < CPUinstance->GetSPLITS(DeviceVars).x) {
        if (j < CPUinstance->GetSPLITS(DeviceVars).y) {
            CPUinstance->BuildLinearSystem(i, j, DeviceVars, DeviceMatrix, DeviceInterim, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ);
        }
    }
    //__syncthreads();
    //if (i < CPUinstance->GetSPLITS(DeviceVars).x * CPUinstance->GetSPLITS(DeviceVars).x) {
    //    if (j < CPUinstance->GetSPLITS(DeviceVars).y * CPUinstance->GetSPLITS(DeviceVars).y) {
    //        CPUinstance->BuildSparseMatrixForSolution(i, j, DeviceVars, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ, PrefixSum);
    //    }
    //}
}

__global__ void TrueMomentumKernel(IncompressCUDA* CPUinstance, double* DeviceVars, double* DevicePSolution, vec3* DeviceMatrix, vec2* DeviceInterim) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = (j * CPUinstance->GetSPLITS(DeviceVars).y) + i;
    double var;
    if (i < CPUinstance->GetSPLITS(DeviceVars).x) {
        if (j < CPUinstance->GetSPLITS(DeviceVars).y) {
            DeviceMatrix[index].p = DevicePSolution[index];
        }
    }
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
    auto LoopStart = std::chrono::high_resolution_clock::now();
    InterimMomentumKernel<<<dim3(1024, 1024), dim3(32, 32)>>> (this, DeviceVars, DeviceMatrix, DeviceInterim);
    auto LoopEnd = std::chrono::high_resolution_clock::now();
    if (debug) {
        std::chrono::duration<double> LOOPTIME = LoopEnd - LoopStart;
        std::cout << "InterimMomentumKernel Execution time -> " << LOOPTIME.count() << " seconds" << std::endl;
        std::cout << "InterimMomentumKernel ->";
    }
}

__host__ void IncompressCUDA::LinearSystemBuilder () {
    auto LoopStart = std::chrono::high_resolution_clock::now();
    LinearSystemKernel<<<dim3(1024, 1024), dim3(32, 32)>>> (this, DeviceVars, DeviceMatrix, DeviceInterim, DeviceRHSVector, DeviceNzCoeffMat, DeviceSparseIndexI, DeviceSparseIndexJ, PrefixSum);
    auto LoopEnd = std::chrono::high_resolution_clock::now();
    if (debug) {
        std::chrono::duration<double> LOOPTIME = LoopEnd - LoopStart;
        std::cout << "LinearSystemKernel Execution time -> " << LOOPTIME.count() << " seconds" << std::endl;
        std::cout << "LinearSystemKernel ->";
    }
}

__host__ void IncompressCUDA::TrueMomentumStep() {
    auto LoopStart = std::chrono::high_resolution_clock::now();
    TrueMomentumKernel << <dim3(1024, 1024), dim3(32, 32)>>> (this, DeviceVars, DevicePSolution, DeviceMatrix, DeviceInterim);
    auto LoopEnd = std::chrono::high_resolution_clock::now();
    if (debug) {
        std::chrono::duration<double> LOOPTIME = LoopEnd - LoopStart;
        std::cout << "TrueMomentumKernel Execution time -> " << LOOPTIME.count() << " seconds" << std::endl;
        std::cout << "TrueMomentumKernel ->";
    }
}

__host__ void IncompressCUDA::SystemDriver() {
    AllocateSystemMatrixMemoryToCPU();
    AllocateInterimMatrixMemoryToCPU();
    AllocateSparseIndexMemoryToCPU();
    AllocateColumnIndexMemoryToCPU();
    AllocateCompressedRowMemoryToCPU();
    AllocateLinearSolutionMemoryToCPU();

    AllocateVariablesMemoryToGPU();
    AllocateMatrixMemoryToGPU();
    AllocateInterimMemoryToGPU();
    AllocateLinearSystemMemoryToGPU();

    //Py_Initialize();

    SetVariablesToGPU();

    std::chrono::duration<double> LOOPTIME, EXECUTETIME, ETA;
    auto init = std::chrono::high_resolution_clock::now();
    for (GetCURRENTSTEP(); GetCURRENTSTEP() < GetSIMSTEPS(); IncreaseSTEP()) {
        auto loopTimer = std::chrono::high_resolution_clock::now();

        if (GetCURRENTSTEP() > 0) {
            //if (CheckConvergedExit()) { break; }
            if (CheckDivergedExit()) { break; }
            else if (!debug) { std::cout << "\033[A\33[2K\r"; }
        }
        else { SetKineticEnergy(); }

        ETA = (LOOPTIME * (((double)GetSIMSTEPS() - 1.000) - (double)GetCURRENTSTEP()));
        auto MINUTES = std::chrono::duration_cast<std::chrono::minutes>(ETA);
        auto HOURS = std::chrono::duration_cast<std::chrono::hours>(MINUTES);
        ETA -= MINUTES;
        MINUTES -= HOURS;

        if (!debug) {
            std::cout << GetCURRENTSTEP() << " / " << GetSIMSTEPS() - 1 << " | Estimated time remaining: ";
            std::cout << HOURS.count() << " Hours ";
            std::cout << MINUTES.count() << " Minutes ";
            std::cout << ETA.count() << " Seconds |" << std::endl;
        }

        SetMatrixToGPU();
        SetInterimToGPU();

        InterimMomentumStep();
        CudaErrorChecker(cudaDeviceSynchronize());
        if (debug) { std::cout << "=====================" << std::endl; }

        SetLinearSystemToGPU();
        LinearSystemBuilder();
        CudaErrorChecker(cudaDeviceSynchronize());
        if (debug) { std::cout << "=====================" << std::endl; }
        GetLinearSystemFromGPU();

        BuildSparseMatrixForSolution();
        FindSparseLinearSolution();
        if (debug) { std::cout << "=====================" << std::endl; }
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

        SetPSolutionToGPU();
        TrueMomentumStep();
        CudaErrorChecker(cudaDeviceSynchronize());
        if (debug) { std::cout << "=====================" << std::endl; }
        GetMatrixFromGPU();
        //cudaDeviceSynchronize();


        //ThrowSystemVariables();
        if (debug) { break; }
        if (GetCURRENTSTEP() % 5 == 0) {
            auto loopEnd = std::chrono::high_resolution_clock::now();
            LOOPTIME = loopEnd - loopTimer;
        }
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