#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
#include <python.h>
#endif

#include "IncompressCUDA.cuh"

__global__ void InterimMomentumKernel(IncompressCUDA* CPUinstance) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 1 && i < CPUinstance->GetSPLITS().x) {
        if (j > 0 && j < CPUinstance->GetSPLITS().y) {
            CPUinstance->ComputeMomentum(i, j, "x");
        }
    }
    if (i > 0 && i < CPUinstance->GetSPLITS().x) {
        if (j > 1 && j < CPUinstance->GetSPLITS().y) {
            CPUinstance->ComputeMomentum(i, j, "x");
        }
    }
}

__global__ void TrueMomentumKernel(IncompressCUDA* CPUinstance) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 1 && i < CPUinstance->GetSPLITS().x) {
        if (j > 0 && j < CPUinstance->GetSPLITS().y) {
            CPUinstance->ComputeIteration(i, j, "x");
        }
    }
    if (i > 0 && i < CPUinstance->GetSPLITS().x) {
        if (j > 1 && j < CPUinstance->GetSPLITS().y) {
            CPUinstance->ComputeIteration(i, j, "x");
        }
    }
}

__host__ void IncompressCUDA::InterimMomentumStep() {
    unsigned int SIZE = (int)GetSPLITS().x * (int)GetSPLITS().y;
    dim3 SPLIT(32, 32, SIZE / 256);
    InterimMomentumKernel<<<SPLIT, SIZE>>> (this);
    return;
}

__host__ void IncompressCUDA::TrueMomentumStep() {
    unsigned int SIZE = (int)GetSPLITS().x * (int)GetSPLITS().y;
    dim3 SPLIT((int)GetSPLITS().x, (int)GetSPLITS().y);
    TrueMomentumKernel<<<SPLIT, SIZE>>> (this);
    std::cout << "Kernel Opened";
}

__host__ void IncompressCUDA::SystemDriver() {
    std::chrono::duration<double> LOOPTIME, EXECUTETIME, ETA;
    auto init = std::chrono::high_resolution_clock::now();

    //AllocateObjectMemoryToGPU();
    AllocateMatrixMemoryToGPU();
    AllocateMatrixSolutionToGPU();
    AllocateInterimMemoryToGPU();
    AllocateInterimSolutionToGPU();
    AllocateLinearSystemMemoryToGPU();

    Py_Initialize();
    for (GetCURRENTSTEP(); GetCURRENTSTEP() < GetSIMSTEPS(); IncreaseSTEP()) {
        auto loopTimer = std::chrono::high_resolution_clock::now();
        ETA = (LOOPTIME * (((double)GetSIMSTEPS() - 1.000) - (double)GetCURRENTSTEP()));
        auto MINUTES = std::chrono::duration_cast<std::chrono::minutes>(ETA);
        auto HOURS = std::chrono::duration_cast<std::chrono::hours>(MINUTES);
        ETA -= MINUTES;
        MINUTES -= HOURS;
        if (GetCURRENTSTEP() > 0) {
            if (CheckConvergedExit()) { break; }
            else if (CheckDivergedExit()) { break; }
            else { std::cout << "\033[A\33[2K\r"; }
        }
        else { SetKineticEnergy(); }
        std::cout << GetCURRENTSTEP() << " / " << GetSIMSTEPS() - 1 << " | Estimated time remaining: ";
        std::cout << HOURS.count() << " Hours ";
        std::cout << MINUTES.count() << " Minutes ";
        std::cout << ETA.count() << " Seconds |" << std::endl;

        SetMatrixToGPU();
        SetInterimToGPU();

        InterimMomentumStep();
        CudaErrorChecker(cudaPeekAtLastError());
        CudaErrorChecker(cudaDeviceSynchronize());
        GetInterimFromGPU(); // CUDA FAILURE

        if (GetCURRENTSTEP() == 0) { break; }

        BuildCoeffMat();
        SetLinearSystemToGPU(); // CUDA FAILURE
        LinearSystemGPUWrapper();
        GetLinearSystemFromGPU(); // CUDA FAILURE
        ReshapeCoefficients();

        ThrowCoefficients();
        ThrowSystemVariables();

        FILE* fd = _Py_fopen("../x64/Debug/SparseSolver.py", "rb");
        PyRun_SimpleFileEx(fd, "SparseSolver.py", 1);

        CatchSolution();

        TrueMomentumStep();
        GetMatrixFromGPU(); // CUDA FAILURE

        ThrowSystemVariables();

        if (GetCURRENTSTEP() % 25 == 0) {
            auto loopEnd = std::chrono::high_resolution_clock::now();
            LOOPTIME = loopEnd - loopTimer;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    EXECUTETIME = end - init;
    Py_Finalize();
    if (!CheckConvergedExit()) { LoopBreakOutput(); }
    else if (!CheckDivergedExit()) { LoopBreakOutput(); }
    std::cout << "System Elapsed Time: " << GetCURRENTSTEP() * GetDT() << " Seconds" << std::endl;
    std::cout << "Loop Execution Time: " << EXECUTETIME.count() << " Seconds" << std::endl;
}