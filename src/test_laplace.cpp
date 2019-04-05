#include "occa.hpp"
#include <iostream>
#include <fstream>
#include <libconfig.h++>
#include <ctime>
#include <string>
#include "EvolveWFC.hpp"

using namespace std;
using namespace occa;
using namespace libconfig;


int main(int argc, char * argv[]) {
    // Setup a GPU device

    int Nx, Ny;
    string mode;
    device dev;

    // Allocate the memory for the Wave function
    float * psi_real, *psi_imag;
    float sigma0, x, y, dx, dy, norm;

    if (argc < 2) {
        cerr << "Error, required at least 1 argument with the cfg file." << endl;
        exit(EXIT_FAILURE);
    }
    float dt;
    LoadCFG(argv[1], Nx, Ny, dx, dy, sigma0, dt, mode);

    dev.setup(mode.c_str());  

    psi_imag = new float[Nx*Ny];
    psi_real = new float[Nx*Ny];

    // Initialize the psi
    norm = sqrtf(sqrtf(2 * M_PI * sigma0*sigma0));
    for (int i = 0; i < Ny; ++i) {
        y = (i - Ny/2) * dy;
        for (int j = 0; j < Nx; ++j) {
            x = (j - Nx/2) * dx;
            psi_imag[Nx*i + j] = 0.f;
            psi_real[Nx*i + j] = expf(- (x*x + y*y) / (2*sigma0*sigma0)) / norm;
        }
    }

    kernel ker;
    ker = dev.buildKernel("laplace2d.okl", "laplace2D");

    // Time the implementation
    time_t start, end;
    start = time(NULL);
    

    // Prepare the memory on the device
    memory GPU_psi_real, GPU_psi_imag;
    GPU_psi_real = dev.malloc(Nx*Ny, dtype::get<float>());
    GPU_psi_imag = dev.malloc(Nx*Ny, dtype::get<float>());

    // Copy the memory from the CPU -> GPU
    GPU_psi_real.copyFrom(psi_real);
    GPU_psi_imag.copyFrom(psi_imag);

    // Load and run the kernel
    ker(Nx, Ny, dx, dy, 1.f, GPU_psi_real, GPU_psi_imag);

    // Copy the laplace operator into the CPU imaginary part
    GPU_psi_imag.copyTo(psi_imag);
    end = time(NULL);

    cout << "Total time for the computation: " << end - start << " s" << endl;

    // Write on a file the result
    ofstream out_psi;
    out_psi.open("D2_psi.dat", ios_base::out);
    out_psi << std::scientific;
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            out_psi << psi_imag[Nx*i + j] << "\t";
        }
        out_psi << endl;
    }
    out_psi.close();

    delete[] psi_real;
    delete[] psi_imag;
    return 0;
}
