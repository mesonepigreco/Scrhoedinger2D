#include <occa.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace occa;


class HilbertSpace {
    public:
    int Nx;
    int Ny;
    float dx;
    float dy;
    float dt;

    float mx;
    float my;
    float k_xx, k_yy, k_xy;

    // The other memory
    float * psi_real, *psi_imag;
    memory o_psi_real;
    memory o_psi_imag;
    device dev;
    kernel ker_laplace;
    kernel ker_scalar;
    kernel ker_harm;


    HilbertSpace(const char * config_file);
    ~HilbertSpace();

    void CopyToGPU();
    void CopyFromGPU();

    void InitGaussian(float sigma_x, float sigma_y, float x0, float y0);

    void LeapFrogGPU(int N_steps);
    void LeapFrogStepGPU();

    void ConfigureFromCFG(const char * config_file);

    /*
     * Perform some measurements on the CPU wavefunction
     * It needs the wavefunction to be already copied on the CPU.
     * You can do that with CopyFromGPU method.
     */
    void MeasureCPU(float &X0, float &Y0, float &sigmaX, float &sigmaY);
};


void LoadCFG(const char * fname, int &Nx, int &Ny, float &dx, float &dy, float &sigma, float &dt, string &mode);