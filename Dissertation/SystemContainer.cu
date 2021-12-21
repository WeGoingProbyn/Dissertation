#include "SystemContainer.cuh"

Container::Container() {
    DT = 0.;
    NU = 0.;
    RE = 0.;
    RHO = 1.;
    CFL = 0.5;
    MAXTIME = 0.0;
    STEPNUMBER = 0;
    BOXSCALE = 0.0;
    SIMSTEPS = 0.0;
    TRUETIME = 0.0;
    TOLERANCE = 1e-8;

    D = vec2();
    SIZE = vec2();
    SPLIT = vec2();
    VelocityBound = vec4();
}

__host__ void Container::AllocateSystemMatrixMemoryToCPU() { SystemMatrix = new vec3[MAXSIZE]; }

__host__ void Container::AllocateInterimMatrixMemoryToCPU() { InterimMatrix = new vec2[MAXSIZE]; }

__host__ void Container::AllocateSparseIndexMemoryToCPU() { SparseIndexesI = new int[NUMNONZEROS]; SparseIndexesJ = new int[NUMNONZEROS]; }

__host__ void Container::AllocateColumnIndexMemoryToCPU() { ColumnIndex = new int[NUMNONZEROS]; }

__host__ void Container::AllocateCompressedRowMemoryToCPU() { RowPointer = new int[MAXSIZE + 1]; }

__host__ void Container::AllocateLinearSolutionMemoryToCPU() { RHSVector = new double[MAXSIZE]; nzCoeffMatrix = new double[NUMNONZEROS]; PSolution = new double[MAXSIZE], HostPrefixSum = new int[NUMNONZEROS]; }

__host__ void Container::DeAllocateSystemMatrixMemoryOnCPU() { delete[] SystemMatrix; }

__host__ void Container::DeAllocateInterimMatrixMemoryOnCPU() { delete[] InterimMatrix; }

__host__ void Container::DeAllocateSparseIndexMemoryToCPU() { delete[] SparseIndexesI; delete[] SparseIndexesJ; }

__host__ void Container::DeAllocateColumnIndexMemoryToCPU() { delete[] ColumnIndex; }

__host__ void Container::DeAllocateCompressedRowMemoryToCPU() { delete[] RowPointer; }

__host__ void Container::DeAllocateLinearSolutionMemoryToCPU() { delete[] RHSVector; delete[] nzCoeffMatrix; delete[] PSolution, delete[] HostPrefixSum; }

__host__ int Container::SetRE(double re) {
    RE = re;
    return 1;
}

__host__ int Container::SetCFL(double cfl) {
    CFL = cfl;
    return 1;
}

__host__ int Container::SetVelocityBoundary(vec4 vbound) {
    VelocityBound.E = vbound.E;
    VelocityBound.W = vbound.W;
    VelocityBound.N = vbound.N;
    VelocityBound.S = vbound.S;
    return 1;
}

__host__ int Container::SetSPLITS(vec2 SPLITS) { SPLIT.x = (int)SPLITS.x; SPLIT.y = (int)SPLITS.y; return 1; }

__host__ int Container::SetSIZE(vec2 SIZES) { SIZE.x = (int)SIZES.x; SIZE.y = (int)SIZES.y; return 1; }

__host__ int Container::SetMAXTIME(double TIME) { MAXTIME = TIME; return 1; }

__host__ void Container::SetAverageVelocities() {
    AverageVelocities = vec2();
    for (int j = 0; j < SPLIT.y; j++) {
        for (int i = 0; i < SPLIT.x; i++) {
            int index = (j * (int)SPLIT.y) + i;
            AverageVelocities.x += SystemMatrix[index].u;
        }
    }
    for (int j = 0; j < SPLIT.y; j++) {
        for (int i = 0; i < SPLIT.x + 1; i++) {
            int index = (j * (int)SPLIT.y) + i;
            AverageVelocities.y += SystemMatrix[index].v;
        }
    }
}

__host__ void Container::SetTOLERANCE(double TOL) { TOLERANCE = TOL; }

__host__ int Container::SetSystemVariables() {
    VBOUND = { VelocityBound.E, VelocityBound.W, VelocityBound.N, VelocityBound.S };

    double splitx = SPLIT.x - 1;
    double splity = SPLIT.y - 1;

    X = std::vector<double>((int)splitx);
    Y = std::vector<double>((int)splity);

    std::generate_n(std::begin(X), (int)splitx, [n = 0, &splitx]() mutable { return n++ / splitx; });
    std::generate_n(std::begin(Y), (int)splity, [n = 0, &splity]() mutable { return n++ / splity; });

    D.x = X[1] - X[0];
    D.y = Y[1] - Y[0];

    RHO = 1.; // Constant rho
    NU = (D.y * SPLIT.x * RHO) / RE;

    DXDY = { D.x, D.y };
    BOXSCALE = (SIZE.x + SIZE.y) / 2 / *std::max_element(std::begin(VBOUND), std::end(VBOUND));
    TRUETIME = BOXSCALE * MAXTIME;
    DT = CFL * *std::min_element(std::begin(DXDY), std::end(DXDY)) / *std::max_element(std::begin(VBOUND), std::end(VBOUND));
    SIMSTEPS = TRUETIME / DT;
    DT = TRUETIME / (int)SIMSTEPS;

    SYSTEMSIZE = (int)SPLIT.x + 1 * (int)SPLIT.y + 1;

    vars[0] = SPLIT.x;
    vars[1] = SPLIT.y;
    vars[2] = D.x;
    vars[3] = D.y;
    vars[4] = DT;
    vars[5] = NU;
    vars[6] = VelocityBound.E;
    vars[7] = VelocityBound.W;
    vars[8] = VelocityBound.N;
    vars[9] = VelocityBound.S;
    return 1;
}

__host__ int Container::IncreaseSTEP() { STEPNUMBER++; return 1; }

__host__ double Container::GetCFL() { return CFL; }

__host__  double Container::GetRE() { return RE; }

__host__  vec4 Container::GetVelocityBoundary() { return VelocityBound; }

__host__  vec2 Container::GetSPLITS() { return SPLIT; }

__host__  vec2 Container::GetSIZE() { return SIZE; }

__host__  vec2 Container::GetD() { return D; }

__host__ double Container::GetMAXTIME() { return MAXTIME; }

__host__ int Container::GetSIMSTEPS() { return (int)SIMSTEPS; }

__host__ int Container::GetCURRENTSTEP() { return STEPNUMBER; }

__host__ double Container::GetTOLERANCE() { return TOLERANCE; }

__host__  double Container::GetDT() { return DT; }

__host__  double Container::GetNU() { return NU; }

__host__ int Container::SetMatrixValue(int i, int j, double var, const char* dim) {
    int index = (j * (int)SPLIT.y) + i;
    if (dim == "u") { SystemMatrix[index].u = var; return 1; }
    else if (dim == "v") { SystemMatrix[index].v = var; return 1; }
    else if (dim == "p") { SystemMatrix[index].p = var; return 1; }
    else { return -1; }
}

__host__ int Container::SetInterimValue(int i, int j, double var, const char* dim) {
    int index = (j * (int)SPLIT.y) + i;
    if (dim == "u") { InterimMatrix[index].x = var; return 1; }
    else if (dim == "v") { InterimMatrix[index].y = var; return 1; }
    else { return -1; }
}

__host__ int Container::SetLinearValue(int i, int j, double var, const char* dim) {
    int TRUEindex = (j * (int)SPLIT.y) + i;
    if (dim == "b") { RHSVector[TRUEindex] = var; return 1; } // Place into AMatrix and BVector not linearsystem
    else if (dim == "a") {
        SparseIndexesI[TRUEindex] = TRUEindex;
        SparseIndexesJ[TRUEindex] = TRUEindex;
        nzCoeffMatrix[TRUEindex] = var;
        return 1;
    }
    else if (dim == "aip") {
        int index = (j * (int)SPLIT.y) + i + ((int)SPLIT.y * (int)SPLIT.x);
        SparseIndexesI[index] = TRUEindex + 1;
        SparseIndexesJ[index] = TRUEindex;
        nzCoeffMatrix[index] = var;
        return 1;
    }
    else if (dim == "ais") { 
        int index = (j * (int)SPLIT.y) + i + (2 * ((int)SPLIT.y * (int)SPLIT.x));
        SparseIndexesI[index] = TRUEindex - 1;
        SparseIndexesJ[index] = TRUEindex;
        nzCoeffMatrix[index] = var;
        return 1; 
    }
    else if (dim == "ajp") {
        int index = (j * (int)SPLIT.y) + i + (3 * ((int)SPLIT.y * (int)SPLIT.x));
        SparseIndexesI[index] = TRUEindex;
        SparseIndexesJ[index] = TRUEindex - (int)SPLIT.x;
        nzCoeffMatrix[index] = var;
        return 1; 
    }
    else if (dim == "ajs") { 
        int index = (j * (int)SPLIT.y) + i + (4 * ((int)SPLIT.y * (int)SPLIT.x));
        SparseIndexesI[index] = TRUEindex;
        SparseIndexesJ[index] = TRUEindex + (int)SPLIT.y;
        nzCoeffMatrix[index] = var;
        return 1; 
    }
    else { return -1; }
}

__host__ void Container::BuildSparseMatrixForSolution() {
    auto LoopStart = std::chrono::high_resolution_clock::now();
    nnz = 5 * (int)GetSPLITS().x * (int)GetSPLITS().y;
    HostPrefixSum[0] = 0;
    for (int i = 1; i < nnz; i++) {
        if (nzCoeffMatrix[i]) { HostPrefixSum[i] = 1; }
        if (!nzCoeffMatrix[i]) { HostPrefixSum[i] = 0; }
    }
    for (int i = 1; i < nnz; i++) {
        HostPrefixSum[i] = HostPrefixSum[i - 1] + HostPrefixSum[i];
    }
    for (int i = 0; i < nnz; i++) {
        if(nzCoeffMatrix[i]) { 
            nzCoeffMatrix[HostPrefixSum[i]] = nzCoeffMatrix[i];
            SparseIndexesI[HostPrefixSum[i]] = SparseIndexesI[i];
            SparseIndexesJ[HostPrefixSum[i]] = SparseIndexesJ[i];
        }
    }
    auto LoopEnd = std::chrono::high_resolution_clock::now();
    if (debug) {
        std::chrono::duration<double> LOOPTIME = LoopEnd - LoopStart;
        std::cout << "BuildSparseMatrixForSolution Execution time -> " << LOOPTIME.count() << " seconds" << std::endl;
        if (debug) { std::cout << "=====================" << std::endl; }
    }
    //for (int i = 0; i < nnz; i++) {
    //    std::cout << "I = " << SparseIndexesI[i] << " , J = " << SparseIndexesJ[i] << " , Value = " << nzCoeffMatrix[i] << " , PrefixSum = " << HostPrefixSum[i] << std::endl;
    //}
}

__host__ void Container::FindSparseLinearSolution() {
    auto LoopStart = std::chrono::high_resolution_clock::now();
    umfpack_di_defaults(Control);
    Control[UMFPACK_PRL] = 3;
    nnz = (5 * (int)GetSPLITS().x * (int)GetSPLITS().y) - (4 * (int)GetSPLITS().y);
    if (debug) { umfpack_di_report_triplet(SPLIT.x * SPLIT.x, SPLIT.y * SPLIT.y, nnz, SparseIndexesJ, SparseIndexesI, nzCoeffMatrix, Control); }
    umfpack_di_triplet_to_col((int)SPLIT.x * (int)SPLIT.x, (int)SPLIT.y * (int)SPLIT.y, nnz, SparseIndexesJ, SparseIndexesI, nzCoeffMatrix, RowPointer, ColumnIndex, nzCoeffMatrix, NULL);
    if (debug) { 
        umfpack_di_report_matrix(SPLIT.x * SPLIT.x, SPLIT.y * SPLIT.y, RowPointer, ColumnIndex, nzCoeffMatrix, 0, Control);
        umfpack_di_report_vector(SPLIT.x * SPLIT.x, RHSVector, Control);
    }
    umfpack_di_symbolic((int)SPLIT.x * (int)SPLIT.x, (int)SPLIT.y * (int)SPLIT.y, RowPointer, ColumnIndex, nzCoeffMatrix, &Symbolic, null, null);
    if (debug) { umfpack_di_report_symbolic(Symbolic, Control); }
    umfpack_di_numeric(RowPointer, ColumnIndex, nzCoeffMatrix, Symbolic, &Numeric, null, null);
    if (debug) { umfpack_di_report_numeric(Numeric, Control); }
    umfpack_di_solve(UMFPACK_A, RowPointer, ColumnIndex, nzCoeffMatrix, PSolution, RHSVector, Numeric, null, null);
    umfpack_di_free_symbolic(&Symbolic);
    umfpack_di_free_numeric(&Numeric);
    auto LoopEnd = std::chrono::high_resolution_clock::now();
    if (debug) {
        std::chrono::duration<double> LOOPTIME = LoopEnd - LoopStart;
        std::cout << "FindSparseLinearSolution Execution time -> " << LOOPTIME.count() << " seconds" << std::endl;
    }
}

__host__ void Container::UpdatePressureValues() {
    for (int i = 0; i < SPLIT.x; i++) {
        for (int j = 0; j < SPLIT.y; j++) {
            int index = (j * (int)SPLIT.y) + i;
            SetMatrixValue(i, j, PSolution[index], "p");
        }
    }
}

__host__ void Container::SetKineticEnergy() {
    KineticEnergy.y = KineticEnergy.x;
    KineticEnergy.x = std::sqrt(std::pow(AverageVelocities.x, 2) + std::pow(AverageVelocities.y, 2));
}

__host__ vec3 Container::GetMatrixValue(int i, int j) { 
    int index = (j * (int)SPLIT.y) + i;
    return SystemMatrix[index];
}

__host__ vec2 Container::GetInterimValue(int i, int j) { 
    int index = (j * (int)SPLIT.y) + i;
    return InterimMatrix[index];
}

__host__ vec2 Container::GetKineticEnergy() { return KineticEnergy; }

__host__ double* Container::GetVariableList() { return vars; }

__host__ vec3* Container::GetSystemMatrix() { return SystemMatrix; }

__host__ vec2* Container::GetInterimMatrix() { return InterimMatrix; }

__host__ int* Container::GetSparseIndexI() { return SparseIndexesI; }

__host__ int* Container::GetSparseIndexJ() { return SparseIndexesJ; }

__host__ int* Container::GetCompressedRowVector() { return RowPointer; }

__host__ int* Container::GetColumnIndexVector() { return ColumnIndex; }

__host__ double* Container::GetRHSVector() { return RHSVector; }

__host__ double* Container::GetnzCoeffMat() { return nzCoeffMatrix; }

__host__ double* Container::GetPSolution() { return PSolution; }

__host__ bool Container::CheckConvergedExit() {
    if (std::abs(GetKineticEnergy().x - GetKineticEnergy().y) < GetTOLERANCE()) { return true; }
    else { return false; }
}

__host__ bool Container::CheckDivergedExit() {
    if (std::abs(GetKineticEnergy().x) > std::numeric_limits<double>::max()) { return true; }
    else { return false; }
}

__host__ void Container::LoopBreakOutput() {
    if (CheckConvergedExit()) {
        std::cout << "Change in system kinetic energy below defined tolerance; System has converged" << std::endl;
    }
    else if (CheckDivergedExit()) {
        std::cout << "Change in system kinetic energy caused overflow; System has diverged" << std::endl;
    }
    std::cout << "<==========================] System Variables: [===========================>" << std::endl;
    std::cout << std::showpoint;
    std::cout << std::setw(20) << "| DX = " << GetD().x << " | DY = " << GetD().y << " | DT = " << GetDT() << " |" << std::endl;
    std::cout << std::noshowpoint;
    std::cout << std::setw(30) << "| SPLITS (x,y) = " << GetSPLITS().x << "," << GetSPLITS().y << "  |  SIZE (x,y) =  " << GetSIZE().x << "," << GetSIZE().y << std::setw(8) << "|" << std::endl;
    std::cout << std::setw(45) << "| Velocity Boundary (T,B,L,R) = " << GetVelocityBoundary().E << "," << GetVelocityBoundary().W << "," << GetVelocityBoundary().N << "," << GetVelocityBoundary().S << std::setw(13) << "|" << std::endl;
    std::cout << std::scientific;
    std::cout << std::setw(47) << "| Kinetic Energy at convergence = " << GetKineticEnergy().x << std::setw(6) << "|" << std::endl;
    std::cout << std::setw(47) << "| Kinetic Energy at iteration-1 = " << GetKineticEnergy().y << std::setw(6) << "|" << std::endl;
    std::cout << std::setw(34) << "| Difference in KE = " << std::abs(GetKineticEnergy().x - GetKineticEnergy().y) << std::setw(19) << "|" << std::endl;
    std::cout << "<==========================================================================>" << std::endl;
    std::cout.unsetf(std::ios::fixed | std::ios::scientific);
}

__host__ int Container::GetSystemVariables() {
    std::cout << "DX = " << D.x << " , ";
    std::cout << "DY = " << D.y << " , ";
    std::cout << "DT = " << DT << std::endl;

    std::cout << "BOXSCALE = " << BOXSCALE << std::endl;
    std::cout << "TRUETIME = " << TRUETIME << std::endl;
    return 1;
}

__host__ int Container::ThrowSystemVariables() {
    std::ofstream SystemFile;
    SystemFile.open("./Output/SystemInfo.txt");
    SystemFile << "ITERATION: " << GetCURRENTSTEP() << std::endl;
    SystemFile << "BOX DIMENSIONS , " << SIZE.x << " , " << SIZE.y << std::endl;
    SystemFile << "MATRIX DIMENSIONS , " << SPLIT.x << " , " << SPLIT.y << std::endl;
    SystemFile << "| X | Y | U | V | P |" << std::endl;
    for (int j = 0; j < SPLIT.y + 1; j++) {
        for (int i = 0; i < SPLIT.x + 1; i++) {
            int index = (j * (int)SPLIT.y) + i;
            SystemFile << (D.x * i) << " , " << (D.y * j) << " , ";
            if (i < SPLIT.x + 1) {
                if (j != SPLIT.y) { 
                    SystemFile << SystemMatrix[index].u << " , ";
                }
                else { SystemFile << "-" << " , "; }
            }
            if (j < SPLIT.y + 1) {
                if (i != SPLIT.x) { 
                    SystemFile << SystemMatrix[index].v << " , ";
                }
                else { SystemFile << "-" << " , "; }
            }
            if (i < SPLIT.x) {
                if (j < SPLIT.y) {
                    SystemFile << SystemMatrix[index].p;
                }
                else { SystemFile << "-"; }
            }
            else { SystemFile << "-"; }
            SystemFile << std::endl;
        }
    }
    SystemFile.close();
    return 1;
};

__host__ void Container::CatchSolution() {
    int it = 0;
    std::string line;
    std::ifstream SolFile("./Output/P_Solution.txt");
    if (SolFile.is_open()) {
        while (std::getline(SolFile, line)) {
            if (it == 0) { ; }
            else { SystemMatrix[it - 1].p = stod(line); }
            it++;
        }
        SolFile.close();
    }
    else { std::cout << "Unable to open file"; }

}