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
    //IncompressCUDA Fluid;
    Fluid.SetRE(1000);
    Fluid.SetCFL(1);
    Fluid.SetVelocityBoundary(vec4(1, 0, 0, 0));
    Fluid.SetSPLITS(vec2(64, 64));
    Fluid.SetSIZE(vec2(1, 1));
    Fluid.SetMAXTIME((double)(60 * 0.2));
    Fluid.SetTOLERANCE((double)1e-9);
    Fluid.SetSystemVariables();

    Fluid.SystemDriver();
    return 0;
}