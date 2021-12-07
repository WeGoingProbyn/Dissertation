#include "FVMPhysics.h"

vec6 Physics::InterpolateVelocities(int i, int j, const char* dim) {
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
    else {
        std::cout << "No dimension found" << std::endl;
        throw - 1;
    }
}

double Physics::ComputeAdvection(int i, int j, const char* dim) {
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
    else {
        std::cout << "No dimension found" << std::endl;
        throw - 1;
    }
}

double Physics::ComputeDiffusion(int i, int j, const char* dim) {
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
    else {
        std::cout << "No dimension found" << std::endl;
        throw - 1;
    }
}

void Physics::ComputeMomentum(int i, int j, const char* dim) {
    double var = (ComputeAdvection(i, j, dim) - ComputeDiffusion(i, j, dim));
    if (dim == "x") { dim = "u"; SetInterimValue(i, j, var, dim); }
    else if (dim == "y") { dim = "v"; SetInterimValue(i, j, var, dim); }
}

void Physics::BuildCoeffMat() {
    LinearSystemMatrix = std::vector<std::vector<dim6>>((int)GetSPLITS().x, std::vector<dim6>((int)GetSPLITS().y));
    LinearSystemRESHAPED = std::vector<dim6>((int)GetSPLITS().x * (int)GetSPLITS().y);
}

void Physics::SetBaseAValues() {
    int POINTS = (int)GetSPLITS().x * (int)GetSPLITS().y;
    for (int i = 0; i < GetSPLITS().x; i++) {
        for (int j = 0; j < GetSPLITS().y; j++) {
            LinearSystemMatrix[i][j].Aipos = -GetDT() * (GetD().y / GetD().x);
            LinearSystemMatrix[i][j].Aisub = -GetDT() * (GetD().y / GetD().x);
            LinearSystemMatrix[i][j].Ajpos = -GetDT() * (GetD().x / GetD().y);
            LinearSystemMatrix[i][j].Ajsub = -GetDT() * (GetD().x / GetD().y);
        }
    }
}

void Physics::BuildTopLeft() {
    int j = 0;
    int i = 0;
    LinearSystemMatrix[i][j].Bvec = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
                                     GetD().y * GetMatrixValue(i, j).u -
                                     GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
                                     GetD().x * GetMatrixValue(i, j).v;
    LinearSystemMatrix[i][j].Acen = (GetDT() * (GetD().y / GetD().x)) + (GetDT() * (GetD().x / GetD().y));
    LinearSystemMatrix[i][j].Aisub = 0;
    LinearSystemMatrix[i][j].Ajpos = 0;
}

void Physics::BuildTopRight() { // Bottom Left
    int j = (int)GetSPLITS().y - 1;
    int i = 0;
    LinearSystemMatrix[i][j].Bvec = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
                                     GetD().y * GetMatrixValue(i, j).u -
                                     GetD().x * GetMatrixValue(i, j + 1).v +
                                     GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y);
    LinearSystemMatrix[i][j].Acen = (GetDT() * (GetD().y / GetD().x)) + GetDT() * (GetD().x / GetD().y);
    LinearSystemMatrix[i][j].Ajsub = 0;
    LinearSystemMatrix[i][j].Aisub = 0;
}

void Physics::BuildBottomLeft() { // Top Right
    int j = 0;
    int i = (int)GetSPLITS().x - 1;
    LinearSystemMatrix[i][j].Bvec = -GetD().y * GetMatrixValue(i + 1, j).u +
                                     GetD().y * (GetMatrixValue(i, j).u + GetDT() * GetInterimValue(i, j).x) -
                                     GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
                                     GetD().x * GetMatrixValue(i, j).v;
    LinearSystemMatrix[i][j].Acen = (GetDT() * (GetD().y / GetD().x)) + (GetDT() * (GetD().x / GetD().y));
    LinearSystemMatrix[i][j].Aipos = 0;
    LinearSystemMatrix[i][j].Ajpos = 0;
}

void Physics::BuildBottomRight() {
    int j = (int)GetSPLITS().y - 1;
    int i = (int)GetSPLITS().x - 1;
    LinearSystemMatrix[i][j].Bvec = -GetD().y * GetMatrixValue(i + 1, j).u +
                                     GetD().y * (GetMatrixValue(j, j).u + GetDT() * GetInterimValue(i, j).x) -
                                     GetD().x * GetMatrixValue(i, j + 1).v +
                                     GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y);
    LinearSystemMatrix[i][j].Acen = (GetDT() * (GetD().y / GetD().x)) + (GetDT() * (GetD().x / GetD().y));
    LinearSystemMatrix[i][j].Ajsub = 0;
    LinearSystemMatrix[i][j].Aipos = 0;
}

void Physics::BuildLeftSide() { // Top side
    int j = 0;
    for (int i = 1; i < GetSPLITS().y - 1; i++) {
        LinearSystemMatrix[i][j].Bvec = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
                                         GetD().y * (GetMatrixValue(i, j).u + GetDT() * GetInterimValue(i, j).x) -
                                         GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
                                         GetD().x * GetMatrixValue(i, j).v; 
        LinearSystemMatrix[i][j].Acen = (2 * GetDT() * (GetD().y / GetD().x)) + (GetDT() * (GetD().x / GetD().y));
        LinearSystemMatrix[i][j].Ajpos = 0;
    }
}

void Physics::BuildRightSide() {
    int j = (int)GetSPLITS().x - 1;
    for (int i = 1; i < GetSPLITS().y - 1; i++) {
        LinearSystemMatrix[i][j].Bvec = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
                                         GetD().y * (GetMatrixValue(i, j).u + GetDT() * GetInterimValue(i, j).x) -
                                         GetD().x * GetMatrixValue(i, j + 1).v +
                                         GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y); 
        LinearSystemMatrix[i][j].Acen = (2 * GetDT() * (GetD().y / GetD().x)) + (GetDT() * (GetD().x / GetD().y));
        LinearSystemMatrix[i][j].Ajsub = 0;
    }
}

void Physics::BuildTopSide() {
    int i = 0;
    for (int j = 1; j < GetSPLITS().x - 1; j++) {
        LinearSystemMatrix[i][j].Bvec = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
                                         GetD().y * GetMatrixValue(i, j).u -
                                         GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
                                         GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y);
        LinearSystemMatrix[i][j].Acen = (GetDT() * (GetD().y / GetD().x)) + (2 * GetDT() * (GetD().x / GetD().y));
        LinearSystemMatrix[i][j].Aisub = 0;
    }
}

void Physics::BuildBottomSide() {
    int i = (int)GetSPLITS().x - 1;
    for (int j = 1; j < GetSPLITS().y - 1; j++) {
        LinearSystemMatrix[i][j].Bvec = -GetD().y * GetMatrixValue(i + 1, j).u +
                                         GetD().y * (GetMatrixValue(i, j).u + GetDT() * GetInterimValue(i, j).x) -
                                         GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
                                         GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y);
        LinearSystemMatrix[i][j].Acen = (GetDT() * (GetD().y / GetD().x)) + (2 * GetDT() * (GetD().x / GetD().y));
        LinearSystemMatrix[i][j].Aipos = 0;
    }
}

void Physics::BuildInterior() {
    for (int j = 1; j < GetSPLITS().x - 1; j++) {
        for (int i = 1; i < GetSPLITS().y - 1; i++) {
            LinearSystemMatrix[i][j].Bvec = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
                                             GetD().y * (GetMatrixValue(i, j).u + GetDT() * GetInterimValue(i, j).x) -
                                             GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
                                             GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y);
            LinearSystemMatrix[i][j].Acen = (2 * GetDT() * (GetD().y / GetD().x)) + (2 * GetDT() * (GetD().x / GetD().y));
        }
    }
}

void Physics::ReshapeCoefficients() {
    int POINTS = (int)GetSPLITS().x * (int)GetSPLITS().y;
    for (int j = 0; j < GetSPLITS().x; j++) {
        for (int i = 0; i < GetSPLITS().y; i++) {
            //std::cout << LinearSystemMatrix[i][j].Bvec << " , ";
            LinearSystemRESHAPED[(j * (int)GetSPLITS().x) + i] = LinearSystemMatrix[i][j];
        }
        //std::cout << std::endl;
    }
}

void Physics::BuildLinearSystem() {
    BuildCoeffMat();
    SetBaseAValues();
    BuildTopLeft();
    BuildTopRight();
    BuildBottomLeft();
    BuildBottomRight();
    BuildLeftSide();
    BuildRightSide();
    BuildTopSide();
    BuildBottomSide();
    BuildInterior();
    ReshapeCoefficients();

    //Acoe = nc::diag<double>(Ajsub, -XSPLIT) + // <-- DON'T DELETE THIS IS IMPORTANT
    //    nc::diag<double>(Aisub, -1) +
    //    nc::diag<double>(Acen, 0) +
    //    nc::diag<double>(Aipos, 1) +
     //   nc::diag<double>(Ajpos, XSPLIT);
    return;
}

void Physics::ComputeIteration(int i, int j, const char* dim) {
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

void Physics::ThrowCoefficients() {
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