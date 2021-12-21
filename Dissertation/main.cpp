#include "Incompressible.h"
#include "IncompressCUDA.cuh"

// Build object
    // Initialise spatial data - DONE
        // Ask user for system variables/System solver - DONE
    // Incorporate Boundary/Initial Conditions - HALF DONE
// Apply mathematical model chosen - DONE
    // "Apply CUDA decomposition" - DONE BASICALLY
    // Update system - DONE
// Post Processing
    // Shit out for python - DONE
    // CUDA beautifier - ?

int main() {
    Incompressible Fluid;
    Fluid.SetRE(1000);
    Fluid.SetCFL(0.5);
    Fluid.SetVelocityBoundary(vec4(1, 0, 0, 0));
    Fluid.SetSPLITS(vec2(32, 32));
    Fluid.SetSIZE(vec2(1, 1));
    Fluid.SetMAXTIME((double)(60 * 1)); // Seconds
    Fluid.SetTOLERANCE((double)1e-16);
    Fluid.SetSystemVariables();

    Fluid.SystemDriver();

    IncompressCUDA CUDA;
    CUDA.SetRE(1000);
    CUDA.SetCFL(0.5);
    CUDA.SetVelocityBoundary(vec4(1, 0, 0, 0));
    CUDA.SetSPLITS(vec2(32, 32));
    CUDA.SetSIZE(vec2(1, 1));
    CUDA.SetMAXTIME((double)(60 * 1)); // Seconds
    CUDA.SetTOLERANCE((double)1e-16);
    CUDA.SetSystemVariables();

    CUDA.SystemDriver();
    return 0;
}