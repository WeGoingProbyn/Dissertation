#include "FVMPhysics.h"

vec6 Physics::InterpolateVelocities(int i, int j, const char* dim) {
    // Find the correct dimension
    if (dim == "x") {
        // These values remain in the domain, no artificial boundary is required
        double UEAST = 0.5 * (GetMatrixValue(i + 1, j).u + GetMatrixValue(i, j).u);
        double UWEST = 0.5 * (GetMatrixValue(i, j).u + GetMatrixValue(i - 1, j).u);
        double VNORTH = 0.5 * (GetMatrixValue(i - 1, j + 1).v + GetMatrixValue(i, j + 1).v);
        double VSOUTH = 0.5 * (GetMatrixValue(i - 1, j).v + GetMatrixValue(i, j).v);
        if (j == 0) {
            double UNORTH = 0.5 * (GetMatrixValue(i, j + 1).u + GetMatrixValue(i, j).u);
            // At the bottom of the domain there is no lower point, an artificial boundary is needed
            double USOUTH = GetVelocityBoundary().E;
            // Return all the information within a 6 component vector structure
            return vec6(UEAST, UWEST, UNORTH, USOUTH, VNORTH, VSOUTH);
        }
        if (j == GetSPLITS().y - 1) {
            double USOUTH = 0.5 * (GetMatrixValue(i, j).u + GetMatrixValue(i, j - 1).u);
            // At the top of the domain there is no higher point, an artificial boundary is needed
            double UNORTH = GetVelocityBoundary().W;
            // Return all the information within a 6 component vector structure
            return vec6(UEAST, UWEST, UNORTH, USOUTH, VNORTH, VSOUTH);
        }
        else {
            // Otherwise there is no bordering boundary and no artificial step is needed
            double UNORTH = 0.5 * (GetMatrixValue(i, j + 1).u + GetMatrixValue(i, j).u);
            double USOUTH = 0.5 * (GetMatrixValue(i, j).u + GetMatrixValue(i, j - 1).u);
            // Return all the information within a 6 component vector structure
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
            double VWEST = 0.5 * (GetMatrixValue(i, j).v + GetMatrixValue(i - 1, j).v);
            double VEAST = GetVelocityBoundary().N;
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
    // Find the dimension
    if (dim == "x") {
        // Interpolate velocities at a given point to the cell walls
        vec6 var1 = InterpolateVelocities(i, j, dim);
        // Find the necessary components laid out in the mathematical theory
        double XX = (var1.E * var1.E - var1.W * var1.W) / GetD().x;
        double XY = (var1.N * var1.EN - var1.S * var1.WS) / GetD().y;
        // Return the final comparison
        return -(XX + XY);
    }

    else if (dim == "y") {
        vec6 var2 = InterpolateVelocities(i, j, dim);
        double YY = (var2.N * var2.N - var2.S * var2.S) / GetD().y;
        double YX = (var2.E * var2.EN - var2.W * var2.WS) / GetD().x;
        return -(YY + YX);
    }
}

double Physics::ComputeDiffusion(int i, int j, const char* dim) {
    // Find the correct dimension
    if (dim == "x") {
        // There is no necessary boundary in these variables as the i value will always lie in the u momentum field
        double XDXe = -2 * GetNU() * (GetMatrixValue(i + 1, j).u - GetMatrixValue(i, j).u) / GetD().x;
        double XDXw = -2 * GetNU() * (GetMatrixValue(i, j).u - GetMatrixValue(i - 1, j).u) / GetD().x;
        if (j == GetSPLITS().y - 1) {
            // At the top of the domain an artificial boundary must be implemented as there is no j value beyond j_max - 1
            double XDYs = -GetNU() * (GetMatrixValue(i, j).u - GetMatrixValue(i, j - 1).u) / GetD().y -
                GetNU() * (GetMatrixValue(i, j).v - GetMatrixValue(i - 1, j).v) / GetD().x;
            double XDYn = -GetNU() * (GetVelocityBoundary().E - GetMatrixValue(i, j).u) / (GetD().y / 2) -
                GetNU() * (GetMatrixValue(i, j + 1).v - GetMatrixValue(i - 1, j + 1).v) / GetD().x;
            return (XDXe - XDXw) / GetD().x + (XDYn - XDYs) / GetD().y;
        }
        if (j == 0) {
            // At the bottom of the domain an artificial boundary must be implemented as there is no j value below 0
            double XDYn = -GetNU() * (GetMatrixValue(i, j + 1).u - GetMatrixValue(i, j).u) / GetD().y -
                GetNU() * (GetMatrixValue(i, j + 1).v - GetMatrixValue(i - 1, j + 1).v) / GetD().x;
            double XDYs = -GetNU() * (GetMatrixValue(i, j).u - GetVelocityBoundary().W) / (GetD().y / 2) -
                GetNU() * (GetMatrixValue(i, j).v - GetMatrixValue(i - 1, j).v) / GetD().x;
            return (XDXe - XDXw) / GetD().x + (XDYn - XDYs) / GetD().y;
        }
        else {
            // Otherwise the relevant comparison can be made, no artificial boundary is necessary
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
    // Find the value of the intermediate computation at a given point
    double var = (ComputeAdvection(i, j, dim) - ComputeDiffusion(i, j, dim));
    // Place the value into the correct intermediate field
    if (dim == "x") { dim = "u"; SetInterimValue(i, j, var, dim); }
    else if (dim == "y") { dim = "v"; SetInterimValue(i, j, var, dim); }
}

void Physics::SetBaseAValues() {
    double var = -GetDT() * (GetD().y / GetD().x);
    for (int i = 0; i < GetSPLITS().x; i++) {
        for (int j = 0; j < GetSPLITS().y; j++) {
            SetLinearValue(i, j, var, "aip");
            SetLinearValue(i, j, var, "ais");
            SetLinearValue(i, j, var, "ajp");
            SetLinearValue(i, j, var, "ajs");
        }
    }
}

void Physics::BuildTopLeft() {
    int j = 0;
    int i = 0;
    double var;
    var = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
           GetD().y * GetMatrixValue(i, j).u -
           GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
           GetD().x * GetMatrixValue(i, j).v;
    SetLinearValue(i, j, var, "b");
    var = (GetDT() * (GetD().y / GetD().x)) + (GetDT() * (GetD().x / GetD().y));
    SetLinearValue(i, j, var, "a");
    SetLinearValue(i, j, 0.0, "ajp"); // No slip
    SetLinearValue(i, j, 0.0, "ais"); // No slip
}

void Physics::BuildTopRight() { // Bottom Left
    int j = (int)GetSPLITS().y - 1;
    int i = 0;
    double var;
    var = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
           GetD().y * GetMatrixValue(i, j).u -
           GetD().x * GetMatrixValue(i, j + 1).v +
           GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y);
    SetLinearValue(i, j, var, "b");
    var = (GetDT() * (GetD().y / GetD().x)) + (GetDT() * (GetD().x / GetD().y));
    SetLinearValue(i, j, var, "a");
    SetLinearValue(i, j, 0.0, "ajs");
    SetLinearValue(i, j, 0.0, "ais");

}

void Physics::BuildBottomLeft() { // Top Right
    int j = 0;
    int i = (int)GetSPLITS().x - 1;
    double var;
    var = -GetD().y * GetMatrixValue(i + 1, j).u +
           GetD().y * (GetMatrixValue(i, j).u + GetDT() * GetInterimValue(i, j).x) -
           GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
           GetD().x * GetMatrixValue(i, j).v;
    SetLinearValue(i, j, var, "b");
    var = (GetDT() * (GetD().y / GetD().x)) + (GetDT() * (GetD().x / GetD().y));
    SetLinearValue(i, j, var, "a");
    SetLinearValue(i, j, 0.0, "ajp");
    SetLinearValue(i, j, 0.0, "aip");
}

void Physics::BuildBottomRight() {
    int j = (int)GetSPLITS().y - 1;
    int i = (int)GetSPLITS().x - 1;
    double var;
    var = -GetD().y * GetMatrixValue(i + 1, j).u +
           GetD().y * (GetMatrixValue(i, j).u + GetDT() * GetInterimValue(i, j).x) -
           GetD().x * GetMatrixValue(i, j + 1).v +
           GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y);
    SetLinearValue(i, j, var, "b");
    var = (GetDT() * (GetD().y / GetD().x)) + (GetDT() * (GetD().x / GetD().y));
    SetLinearValue(i, j, var, "a");
    SetLinearValue(i, j, 0.0, "ajs");
    SetLinearValue(i, j, 0.0, "aip");
}

void Physics::BuildLeftSide() { // Top side
    int j = 0;
    double var;
    for (int i = 1; i < GetSPLITS().y - 1; i++) {
        var = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
            GetD().y * (GetMatrixValue(i, j).u + GetDT() * GetInterimValue(i, j).x) -
            GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
            GetD().x * GetMatrixValue(i, j).v;
        SetLinearValue(i, j, var, "b");
        var = (2 * GetDT() * (GetD().y / GetD().x)) + (GetDT() * (GetD().x / GetD().y));
        SetLinearValue(i, j, var, "a");
        SetLinearValue(i, j, 0.0, "ajp");
    }
}

void Physics::BuildRightSide() {
    int j = (int)GetSPLITS().x - 1;
    double var;
    for (int i = 1; i < GetSPLITS().y - 1; i++) {
        var = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
            GetD().y * (GetMatrixValue(i, j).u + GetDT() * GetInterimValue(i, j).x) -
            GetD().x * GetMatrixValue(i, j + 1).v +
            GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y);
        SetLinearValue(i, j, var, "b");
        var = (2 * GetDT() * (GetD().y / GetD().x)) + (GetDT() * (GetD().x / GetD().y));
        SetLinearValue(i, j, var, "a");
        SetLinearValue(i, j, 0.0, "ajs");
    }
}

void Physics::BuildTopSide() {
    int i = 0;
    double var;
    for (int j = 1; j < GetSPLITS().x - 1; j++) {
        var = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
            GetD().y * GetMatrixValue(i, j).u -
            GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
            GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y);
        SetLinearValue(i, j, var, "b");
        var = (GetDT() * (GetD().y / GetD().x)) + (2 * GetDT() * (GetD().x / GetD().y));
        SetLinearValue(i, j, var, "a");
        //if (!tunnel) {
        SetLinearValue(i, j, 0.0, "ais");
        //}
    }
}

void Physics::BuildBottomSide() {
    int i = (int)GetSPLITS().x - 1;
    double var;
    for (int j = 1; j < GetSPLITS().y - 1; j++) {
        var = -GetD().y * GetMatrixValue(i + 1, j).u +
               GetD().y * (GetMatrixValue(i, j).u + GetDT() * GetInterimValue(i, j).x) -
               GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
               GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y);
        SetLinearValue(i, j, var, "b");
        var = (GetDT() * (GetD().y / GetD().x)) + (2 * GetDT() * (GetD().x / GetD().y));
        SetLinearValue(i, j, var, "a");
        //if (!tunnel) {
        SetLinearValue(i, j, 0.0, "aip");
        //}
    }
}

void Physics::BuildInterior() {
    // Declare variable
    double var;
    // Need to loop over all cells which don't border a boundary
    for (int j = 1; j < GetSPLITS().x - 1; j++) {
        for (int i = 1; i < GetSPLITS().y - 1; i++) {
            // Find the value for the B vector using the theories derived
            var = -GetD().y * (GetMatrixValue(i + 1, j).u + GetDT() * GetInterimValue(i + 1, j).x) +
                   GetD().y * (GetMatrixValue(i, j).u + GetDT() * GetInterimValue(i, j).x) -
                   GetD().x * (GetMatrixValue(i, j + 1).v + GetDT() * GetInterimValue(i, j + 1).y) +
                   GetD().x * (GetMatrixValue(i, j).v + GetDT() * GetInterimValue(i, j).y);
            // Set to the correct part of the linear system matrix
            SetLinearValue(i, j, var, "b");
            // Find the necessary coefficients for the coefficients matrix
            var = (2 * GetDT() * (GetD().y / GetD().x)) + (2 * GetDT() * (GetD().x / GetD().y));
            // Set to the correct part of the linear system matrix
            SetLinearValue(i, j, var, "a");
        }
    }
    if (debug) {
        if (GetSPLITS().x < 17) {
            for (int j = 0; j < GetSPLITS().x; j++) {
                for (int i = 0; i < GetSPLITS().y; i++) {
                    std::cout << GetLinearValue(i, j, "b") << " , ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            for (int j = 0; j < GetSPLITS().x; j++) {
                for (int i = 0; i < GetSPLITS().y; i++) {
                    std::cout << GetLinearValue(i, j, "a") << " , ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            for (int j = 0; j < GetSPLITS().x; j++) {
                for (int i = 0; i < GetSPLITS().y; i++) {
                    std::cout << GetLinearValue(i, j, "ais") << " , ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            for (int j = 0; j < GetSPLITS().x; j++) {
                for (int i = 0; i < GetSPLITS().y; i++) {
                    std::cout << GetLinearValue(i, j, "aip") << " , ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            for (int j = 0; j < GetSPLITS().x; j++) {
                for (int i = 0; i < GetSPLITS().y; i++) {
                    std::cout << GetLinearValue(i, j, "ajs") << " , ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            for (int j = 0; j < GetSPLITS().x; j++) {
                for (int i = 0; i < GetSPLITS().y; i++) {
                    std::cout << GetLinearValue(i, j, "ajp") << " , ";
                }
                std::cout << std::endl;
            }
        }
    }
}

void Physics::BuildLinearSystem() {
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
    //if (debug) {
    //    if (GetSPLITS().x < 6) {
    //    }
    //}
    return;
}

void Physics::ComputeIteration(int i, int j, const char* dim) {
    // Find which dimension is needed
    if (dim == "x") { 
        dim = "u"; 
        // Find the new variable at a given cell using intermediate momentum and pressure field
        double var = GetMatrixValue(i, j).u + (GetDT() * GetInterimValue(i, j).x) - GetDT() * 
                     (GetMatrixValue(i, j).p - GetMatrixValue(i - 1, j).p) / GetD().x;
        // Set the new value to the system matrix and the given point
        SetMatrixValue(i, j, var, dim);
    }
    else if (dim == "y") {
        dim = "v";
        double var = GetMatrixValue(i, j).v + (GetDT() * GetInterimValue(i, j).y) - GetDT() * (GetMatrixValue(i, j).p - GetMatrixValue(i, j - 1).p) / GetD().y;
        SetMatrixValue(i, j, var, dim);
    }
}

void Physics::CheckBoundaryCell(int i, int j) {
    if (shape) {
        if (CheckBoundaryCondition(i, j, circle) < 2e-16) { // Boundary Conditions 
            const char* dim = "u";
            SetMatrixValue(i, j, 0.0, dim);
            SetMatrixValue(i + 1, j, 0.0, dim);
            SetMatrixValue(i - 1, j, 0.0, dim);
            SetMatrixValue(i, j + 1, 0.0, dim);
            SetMatrixValue(i, j - 1, 0.0, dim);
            const char* dim_ = "v";
            SetMatrixValue(i, j, 0.0, dim_);
            SetMatrixValue(i + 1, j, 0.0, dim_);
            SetMatrixValue(i - 1, j, 0.0, dim_);
            SetMatrixValue(i, j + 1, 0.0, dim_);
            SetMatrixValue(i, j - 1, 0.0, dim_);
            const char* _dim = "p";
            SetMatrixValue(i, j, 0.0, _dim);
            SetMatrixValue(i + 1, j, GetMatrixValue(i + 1, j).p - (1 * GetNU() / GetD().y) * (-2 * GetMatrixValue(i + 1, j).u + GetMatrixValue(i + 2, j).u), _dim);
            SetMatrixValue(i - 1, j, GetMatrixValue(i - 1, j).p - (1 * GetNU() / GetD().y) * (-2 * GetMatrixValue(i - 2, j).u + GetMatrixValue(i - 3, j).u), _dim);
            SetMatrixValue(i, j + 1, GetMatrixValue(i, j + 1).p - (1 * GetNU() / GetD().y) * (-2 * GetMatrixValue(i, j + 1).v + GetMatrixValue(i, j + 2).v), _dim);
            SetMatrixValue(i, j - 1, GetMatrixValue(i, j - 1).p - (1 * GetNU() / GetD().y) * (-2 * GetMatrixValue(i, j - 2).v + GetMatrixValue(i, j - 3).v), _dim);
        }
    }
    else { return; }
}

/*void Physics::ThrowCoefficients() {
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