#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
#include <python.h>
#endif

#include "Incompressible.h"

void Incompressible::InterimMomentumStep() {
    for (int i = 1; i < GetSPLITS().x; i++) {
        for (int j = 0; j < GetSPLITS().y; j++) {
            ComputeMomentum(i, j, "x");
        }
    }
    for (int i = 0; i < GetSPLITS().y; i++) {
        for (int j = 1; j < GetSPLITS().x; j++) {
            ComputeMomentum(i, j, "y");
            //std::cout << GetInterimValue(j, i).y << " , ";
        }
        //std::cout << std::endl;
    }
    return;
}

void Incompressible::TrueMomentumStep() {
    for (int i = 1; i < GetSPLITS().x; i++) {
        for (int j = 0; j < GetSPLITS().y; j++) {
            ComputeIteration(i, j, "x");
            //std::cout << GetMatrixValue(j, i).u << " , ";
        }
        //std::cout << std::endl;
    }
    for (int i = 0; i < GetSPLITS().y; i++) {
        for (int j = 1; j < GetSPLITS().x; j++) {
            ComputeIteration(i, j, "y");
        }
    }
    SetAverageVelocities();
    SetKineticEnergy();
}

void Incompressible::SystemDriver() {
    Py_Initialize();
    std::chrono::duration<double> LOOPTIME, EXECUTETIME, ETA;
    auto init = std::chrono::high_resolution_clock::now();
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

        InterimMomentumStep();
        BuildLinearSystem();
        ThrowCoefficients();
        ThrowSystemVariables();

        FILE* fd = _Py_fopen("../x64/Debug/SparseSolver.py", "rb");
        PyRun_SimpleFileEx(fd, "SparseSolver.py", 1);

        //if (GetCURRENTSTEP() == 1) { break; }
        CatchSolution();
        TrueMomentumStep();
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