#include "Incompressible.h"
#include "IncompressCUDA.cuh"

int main() {
    Incompressible Fluid;
    Fluid.SetFlags(true, false, false);
    Fluid.SetRE(1000);
    Fluid.SetCFL(0.5);
    Fluid.SetVelocityBoundary(vec4(1, 0, 0, 0));
    Fluid.SetSPLITS(vec2(129, 129));
    Fluid.SetSIZE(vec2(1, 1));
    Fluid.SetMAXTIME((double)(60 * 10)); // Seconds
    Fluid.SetTOLERANCE((double)1e-8);
    Fluid.SetSystemVariables();
    //Fluid.SetObjectArguments(vec4(32, 32, 16, false));
    Fluid.SetObjectArguments(vec4(64, 64, 32, false));

    Fluid.SystemDriver();
    system("cd ../x64/Debug && python Grapher.py");
    system("start Output");

    //IncompressCUDA CUDA;
    //CUDA.SetRE(1000);
    //CUDA.SetCFL(0.5);
    //CUDA.SetVelocityBoundary(vec4(1, 0, 0, 0));
    //CUDA.SetSPLITS(vec2(32, 32));
    //CUDA.SetSIZE(vec2(1, 1));
    //CUDA.SetMAXTIME((double)(60 * 1)); // Seconds
    //CUDA.SetTOLERANCE((double)1e-16);
    //CUDA.SetSystemVariables();

    //CUDA.SystemDriver();
    return 0;
}