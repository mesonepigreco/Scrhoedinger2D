#include <occa.hpp>
#include <iostream>
#include <string>

#define NMAXITER 10

using namespace std;
using namespace occa;

#define ALGORITHM_LF "Leap-Frog"
#define ALGORITHM_CN "Crank-Nicolson"



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

    string algorithm;

    // The other memory
    float * psi_real, *psi_imag;
    memory o_psi_real, o_psi_re_tmp;
    memory o_psi_imag, o_psi_imag_tmp;
    memory o_work;
    device dev;
    kernel ker_laplace;
    kernel ker_scalar;
    kernel ker_harm;
    kernel ker_refresh;
    kernel ker_sum;


    HilbertSpace(const char * config_file);
    ~HilbertSpace();

    void CopyToGPU();
    void CopyFromGPU();

    void InitGaussian(float sigma_x, float sigma_y, float x0, float y0);

    /* 
     * Evolve the system on the GPU using the LeapFrog algorithm
     */
    void LeapFrogGPU(int N_steps);

    // Evolve the system on the GPU using the Crank-Nicolson algorithm
    void CNGPU(int N_steps);

    // Evolve according to the algorithm that satisfy the key
    // Satisfying the algorithm keyword
    void EvolveFunctionGPU(int Nsteps);

    void LeapFrogStepGPU();
    void ApplyInverseHalfEulerStep(memory psi_real, memory psi_imag, int Ntimes);
    void ApplyCayleyOperator();
    

    void ConfigureFromCFG(const char * config_file);


    /*
     * Perform some measurements on the CPU wavefunction
     * It needs the wavefunction to be already copied on the CPU.
     * You can do that with CopyFromGPU method.
     */
    void MeasureCPU(float &X0, float &Y0, float &sigmaX, float &sigmaY, float &norm);
};


void LoadCFG(const char * fname, int &Nx, int &Ny, float &dx, float &dy, float &sigma, float &dt, string &mode);