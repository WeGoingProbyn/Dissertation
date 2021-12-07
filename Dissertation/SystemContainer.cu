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
            AverageVelocities.x += SystemMatrix[i][j].u;
        }
    }
    for (int j = 0; j < SPLIT.y; j++) {
        for (int i = 0; i < SPLIT.x + 1; i++) {
            AverageVelocities.y += SystemMatrix[i][j].v;
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

    SystemMatrix = std::vector<std::vector<vec3>>((int)SPLIT.x + 1, std::vector<vec3>((int)SPLIT.y + 1));
    InterimMatrix = std::vector<std::vector<vec2>>((int)SPLIT.x + 1, std::vector<vec2>((int)SPLIT.y + 1));
    return 1;
} // BROKE

__host__ int Container::IncreaseSTEP() { STEPNUMBER++; return 1; }

__host__ double Container::GetCFL() { return CFL; }

__host__ __device__ double Container::GetRE() { return RE; }

__host__ __device__ vec4 Container::GetVelocityBoundary() { return VelocityBound; }

__host__ __device__ vec2 Container::GetSPLITS() { return SPLIT; }

__host__ __device__ vec2 Container::GetSIZE() { return SIZE; }

__host__ __device__ vec2 Container::GetD() { return D; }

__host__ double Container::GetMAXTIME() { return MAXTIME; }

__host__ int Container::GetSIMSTEPS() { return (int)SIMSTEPS; }

__host__ int Container::GetCURRENTSTEP() { return STEPNUMBER; }

__host__ double Container::GetTOLERANCE() { return TOLERANCE; }

__host__ __device__ double Container::GetDT() { return DT; }

__host__ __device__ double Container::GetNU() { return NU; }

__host__ int Container::SetMatrixValue(int i, int j, double var, const char* dim) {
    if (dim == "u") { SystemMatrix[i][j].u = var; return 1; }
    else if (dim == "v") { SystemMatrix[i][j].v = var; return 1; }
    else if (dim == "p") { SystemMatrix[i][j].p = var; return 1; }
    else { return -1; }
}

__host__ int Container::SetInterimValue(int i, int j, double var, const char* dim) {
    if (dim == "u") { InterimMatrix[i][j].x = var; return 1; }
    else if (dim == "v") { InterimMatrix[i][j].y = var; return 1; }
    else { return -1; }
}

__host__ void Container::SetKineticEnergy() {
    KineticEnergy.y = KineticEnergy.x;
    KineticEnergy.x = std::sqrt(std::pow(AverageVelocities.x, 2) + std::pow(AverageVelocities.y, 2));
}

__host__ vec3 Container::GetMatrixValue(int i, int j) { return SystemMatrix[i][j]; }

__host__ vec2 Container::GetInterimValue(int i, int j) { return InterimMatrix[i][j]; }

__host__ vec2 Container::GetKineticEnergy() { return KineticEnergy; }

__host__ std::vector<std::vector<vec3>> Container::GetSystemMatrix() { return SystemMatrix; }

__host__ std::vector<std::vector<vec2>> Container::GetInterimMatrix() { return InterimMatrix; }

__host__ bool Container::CheckConvergedExit() {
    if (std::abs(GetKineticEnergy().x - GetKineticEnergy().y) < GetTOLERANCE()) { return true; } // Can maybe use this to format console output
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
            //std::cout << InterimMatrix[i][j].x << " , ";
            SystemFile << (D.x * i) << " , " << (D.y* j) << " , ";
            if (i < SPLIT.x + 1) {
                if (j != SPLIT.y) { SystemFile << SystemMatrix[i][j].u << " , "; }
                else { SystemFile << "-" << " , "; }
            }
            if (j < SPLIT.y + 1) {
                if (i != SPLIT.x) { SystemFile << SystemMatrix[i][j].v << " , "; }
                else { SystemFile << "-" << " , "; }
            }
            if (i < SPLIT.x) {
                if (j < SPLIT.y) { SystemFile << SystemMatrix[i][j].p; }
                else { SystemFile << "-"; }
            }
            else { SystemFile << "-"; }
            SystemFile << std::endl;
        }
        //std::cout << std::endl;
    }
    //std::cout << std::endl;
    SystemFile.close();
    return 1;
};

__host__ void Container::CatchSolution() {
    int it = 0;
    std::string line;
    std::ifstream SolFile("./Output/P_Solution.txt");
    P_SOL = std::vector<double>((unsigned int)SPLIT.x * (unsigned int)SPLIT.y);
    if (SolFile.is_open()) {
        while (std::getline(SolFile, line)) {
            if (it == 0) { ; }
            else { P_SOL[it - 1] = stod(line); }
            it++;
        }
        for (int i = 0; i < SPLIT.x; i++) {
            for (int j = 0; j < SPLIT.y; j++) {
                SystemMatrix[i][j].p = P_SOL[(unsigned int)(j * (unsigned int)SPLIT.x) + i];
            }
        }
        SolFile.close();
    }
    else { std::cout << "Unable to open file"; }

}