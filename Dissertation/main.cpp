#include "Incompressible.h"
//#include "IncompressCUDA.cuh"

// Build object
    // Initialise spatial data
        // Ask user for system variables/System solver
    // Incorporate Boundary/Initial Conditions
// Apply mathematical model chosen
    // "Apply CUDA decomposition"	
    // Update system
// Post Processing
    // Shit out for python
    // CUDA beautifier

int main() {
    Incompressible Fluid;
    Fluid.SetRE(10000);
    Fluid.SetCFL(0.1);
    Fluid.SetVelocityBoundary(vec4(1, 0, -2, 0));
    Fluid.SetSPLITS(vec2(64, 64));
    Fluid.SetSIZE(vec2(1, 1));
    Fluid.SetMAXTIME((double)(60*5));
    Fluid.SetTOLERANCE((double)1e-16);
    Fluid.SetSystemVariables();

    Fluid.SystemDriver();
    Fluid.ThrowSystemVariables();

    /*IncompressCUDA CUDA;
    CUDA.SetRE(100);
    CUDA.SetCFL(0.5);
    CUDA.SetVelocityBoundary(vec4(1, 0, 0, 0));
    CUDA.SetSPLITS(vec2(8, 8));
    CUDA.SetSIZE(vec2(1, 1));
    CUDA.SetMAXTIME((double)(60 * 0.1));
    CUDA.SetSystemVariables();

    CUDA.SystemDriver();
    CUDA.ThrowSystemVariables();
    return 0;*/
}