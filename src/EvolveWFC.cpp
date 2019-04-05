#include "EvolveWFC.hpp"
#include <libconfig.h++>


using namespace libconfig;

void LoadCFG(const char * fname, int &Nx, int &Ny, float &dx, float &dy, float &sigma, float &dt, string &mode) {
    Config cfg;

    try {
        cfg.readFile(fname);
    } catch (const FileIOException &e) {
        cerr << "Error, file " << fname << " not found." << endl;
        throw;
    } catch (const ParseException &e) {
        cerr << "Parse error at " << e.getFile() << " : " << e.getLine() << endl;
        cerr << e.getError() << endl;
        throw;
    }

    mode = "mode: 'Serial'";
    dt = 1.0;
    try {
        Nx = cfg.lookup("Nx");
        Ny = cfg.lookup("Ny");
        dx = cfg.lookup("dx");
        dy = cfg.lookup("dy");
        sigma = cfg.lookup("sigma");
        if (!cfg.lookupValue("dt", dt)) {
            cout << "dt" << " not found in the input file." << endl;
            cout << "I assume dt = " << dt << endl;
        }
        if (!cfg.lookupValue("mode", mode)) {
            cout << "mode, not found in input file" << endl;
            cout << "Using a serial algorithm." << endl;
        }
    } catch (const SettingNotFoundException &e) {
        cerr << "Error, setting " << e.getPath() << " not found." << endl;
        cerr << e.what() << endl;
        throw;
    } catch (const SettingTypeException &e) {
        cerr << "Error, wrong type for setting : " << e.getPath() << endl;
        cerr << e.what() << endl;
        throw;
    }
}

HilbertSpace::HilbertSpace(const char * config_file) {
    string mode;
    float sigma;
    LoadCFG(config_file, Nx, Ny, dx, dy, sigma, dt, mode);

    // Prepare the device
    dev.setup(mode.c_str());
    mx = 1.f;
    my = 1.f;

    // Prepare the host and device memory
    o_psi_imag = dev.malloc(Nx*Ny, occa::dtype::get<float>());
    o_psi_real = dev.malloc(Nx*Ny, occa::dtype::get<float>());
    
    psi_real = new float[Nx*Ny];
    psi_imag = new float[Nx*Ny];

    // Setup the kernels
    ker_laplace = dev.buildKernel("laplace2d.okl", "laplace2D");
    ker_scalar = dev.buildKernel("laplace2d.okl", "scalardot");
    ker_harm = dev.buildKernel("laplace2d.okl", "ApplyHarmonicHamiltonian");

    // Initialize as a gaussian
    InitGaussian(sigma, sigma, 0, 0);
}

HilbertSpace::~HilbertSpace() {
    delete[] psi_real;
    delete[] psi_imag;
}

void HilbertSpace::CopyToGPU() {
    o_psi_imag.copyFrom(psi_imag);
    o_psi_real.copyFrom(psi_real);
}

void HilbertSpace::CopyFromGPU() {
    o_psi_imag.copyTo(psi_imag);
    o_psi_real.copyTo(psi_real);
}

void HilbertSpace::InitGaussian(float sigma_x, float sigma_y, float x0, float y0) {
    float norm;

    norm = 1.f/ sqrtf(sqrtf(2*M_PI)*sigma_x * sigma_y);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            float x = (i - Ny/2) * dx;
            float y = (j - Nx/2) * dx;
            psi_real[Nx*i + j] = expf(- (x - x0)*(x-x0)/(2*sigma_x*sigma_x) - (y -y0)*(y-y0)/(2*sigma_y*sigma_y));
            psi_real[Nx*i + j] /= norm;
            psi_imag[Nx*i + j] = 0.f;
        }
    }
}

void HilbertSpace::LeapFrogStepGPU() {
    // Advance the imaginary part before the real one
    ker_laplace(Nx, Ny, dx*mx, dy*my, dt/2, o_psi_real, o_psi_imag);
    ker_harm(Nx, Ny, dx, dy, 0, 0, k, -dt, o_psi_real, o_psi_imag);

    // Advance the real part
    ker_laplace(Nx, Ny, dx*mx, dy*my, -dt/2, o_psi_imag, o_psi_real);
    ker_harm(Nx, Ny, dx, dy, 0, 0, k, dt, o_psi_imag, o_psi_real);
}


void HilbertSpace::LeapFrogGPU(int N_steps) {
    for (int i = 0; i < N_steps; ++i) {
        LeapFrogStepGPU();
    }
}


void HilbertSpace::MeasureCPU(float &X0, float &Y0, float &sigmaX, float &sigmaY) {
    X0 = 0;
    Y0 = 0;
    sigmaX = 0;
    sigmaY = 0;

    #pragma omp parallel for collapse(2) reduction(+:X0,Y0,sigmaX,sigmaY)
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            float x = (j - Nx/2) * dx;
            float y = (i - Ny/2) * dy;
            float p = psi_real[Nx*i+j]*psi_real[Nx*i+j] + psi_imag[Nx*i+j]*psi_imag[Nx*i+j];
            p *= dx*dy;
            X0 += p*x;
            Y0 += p*y;
            sigmaX += p*x*x;
            sigmaY += p*y*y;
        }
    }
    sigmaX -= X0*X0;
    sigmaY -= Y0*Y0;
}