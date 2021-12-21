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
        }
    }
    return;
}

void Incompressible::TrueMomentumStep() {
    for (int i = 1; i < GetSPLITS().x; i++) {
        for (int j = 0; j < GetSPLITS().y; j++) {
            ComputeIteration(i, j, "x");
        }
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
    AllocateSystemMatrixMemoryToCPU();
    AllocateInterimMatrixMemoryToCPU();
    AllocateSparseIndexMemoryToCPU();
    AllocateColumnIndexMemoryToCPU();
    AllocateCompressedRowMemoryToCPU();
    AllocateLinearSolutionMemoryToCPU();

    std::chrono::duration<double> LOOPTIME, EXECUTETIME, ETA;
    auto init = std::chrono::high_resolution_clock::now();
    for (GetCURRENTSTEP(); GetCURRENTSTEP() < GetSIMSTEPS(); IncreaseSTEP()) {
        auto loopTimer = std::chrono::high_resolution_clock::now();

        if (GetCURRENTSTEP() > 0) {
            if (CheckConvergedExit()) { break; }
            else if (CheckDivergedExit()) { break; }
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

        if (debug) { std::cout << "=====================" << std::endl; }
        auto InterimStart = std::chrono::high_resolution_clock::now();
        InterimMomentumStep();
        auto InterimEnd = std::chrono::high_resolution_clock::now();
        if (debug) {
            std::chrono::duration<double> LOOPTIME = InterimEnd - InterimStart;
            std::cout << "InterimMomentumStep Execution time -> " << LOOPTIME.count() << " seconds" << std::endl;
            std::cout << "=====================" << std::endl;
        }


        auto LinearStart = std::chrono::high_resolution_clock::now();
        BuildLinearSystem();
        auto LinearEnd = std::chrono::high_resolution_clock::now();
        if (debug) {
            std::chrono::duration<double> LOOPTIME = LinearEnd - LinearStart;
            std::cout << "BuildLinearSystem Execution time -> " << LOOPTIME.count() << " seconds" << std::endl;
            std::cout << "=====================" << std::endl;
        }

        BuildSparseMatrixForSolution();

        FindSparseLinearSolution();
        if (debug) { std::cout << "=====================" << std::endl; }

        auto PreStart = std::chrono::high_resolution_clock::now();
        UpdatePressureValues();
        auto PreEnd = std::chrono::high_resolution_clock::now();
        if (debug) {
            std::chrono::duration<double> LOOPTIME = PreEnd - PreStart;
            std::cout << "UpdatePressureValues Execution time -> " << LOOPTIME.count() << " seconds" << std::endl;
            std::cout << "=====================" << std::endl;
        }

        

        auto TrueStart = std::chrono::high_resolution_clock::now();
        TrueMomentumStep();
        auto TrueEnd = std::chrono::high_resolution_clock::now();
        if (debug) {
            std::chrono::duration<double> LOOPTIME = TrueEnd - TrueStart;
            std::cout << "TrueMomentumStep Execution time -> " << LOOPTIME.count() << " seconds" << std::endl;
            std::cout << "=====================" << std::endl;
        }

        if (debug) { break; }

        if (GetCURRENTSTEP() % 50 == 0) {
            auto loopEnd = std::chrono::high_resolution_clock::now();
            LOOPTIME = loopEnd - loopTimer;
        }
        //if (GetCURRENTSTEP() % 100 == 0) {
        //    ThrowSystemVariables();
        //}

    }
    if (!CheckConvergedExit()) { LoopBreakOutput(); }
    else if (!CheckDivergedExit()) { LoopBreakOutput(); }

    auto end = std::chrono::high_resolution_clock::now();
    EXECUTETIME = end - init;

    std::cout << "System Elapsed Time: " << GetCURRENTSTEP() * GetDT() << " Seconds" << std::endl;
    std::cout << "Loop Execution Time: " << EXECUTETIME.count() << " Seconds" << std::endl;

    ThrowSystemVariables();

    DeAllocateSystemMatrixMemoryOnCPU();
    DeAllocateInterimMatrixMemoryOnCPU();
    DeAllocateSparseIndexMemoryToCPU();
    DeAllocateColumnIndexMemoryToCPU();
    DeAllocateCompressedRowMemoryToCPU();
    DeAllocateLinearSolutionMemoryToCPU();
}