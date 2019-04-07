#include "EvolveWFC.hpp"
#include <libconfig.h++>

// We include the binary version of the kernel string
// Parsed by the xxd utility.
// This enable to correctly include the okl kernel
#include "laplace2d.okbin"

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
    string kernel_source = reinterpret_cast<char*>(src_laplace2d_okl);

    //cout << "Checking the kernel source:" << endl;
    //cout << kernel_source << endl;

    // Prepare the device
    dev.setup(mode.c_str());
    mx = 1.f;
    my = 1.f;

    // Prepare the host and device memory
    o_psi_imag = dev.malloc(Nx*Ny, occa::dtype::get<float>());
    o_psi_real = dev.malloc(Nx*Ny, occa::dtype::get<float>());
    o_work = dev.malloc(Nx*Ny, occa::dtype::get<float>());
    
    psi_real = new float[Nx*Ny];
    psi_imag = new float[Nx*Ny];

    // Setup the kernels
    ker_laplace = dev.buildKernelFromString(kernel_source, "laplace2D");
    ker_scalar = dev.buildKernelFromString(kernel_source, "scalardot");
    ker_harm = dev.buildKernelFromString(kernel_source, "ApplyHarmonicHamiltonian");
    ker_refresh = dev.buildKernelFromString(kernel_source, "refresh");
    ker_sum = dev.buildKernelFromString(kernel_source, "refresh");
    
    //ker_laplace = dev.buildKernel("laplace2d.okl", "laplace2D");
    //ker_scalar = dev.buildKernel("laplace2d.okl", "scalardot");
    //ker_harm = dev.buildKernel("laplace2d.okl", "ApplyHarmonicHamiltonian");

    ConfigureFromCFG(config_file);

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

    norm = sqrtf(M_PI*sigma_x * sigma_y);
    float test = 0;

    #pragma omp parallel for collapse(2) reduction(+:test)
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            float x = (i - Ny/2) * dx;
            float y = (j - Nx/2) * dx;
            psi_real[Nx*i + j] = expf(- (x - x0)*(x-x0)/(2*sigma_x*sigma_x) - (y -y0)*(y-y0)/(2*sigma_y*sigma_y));
            psi_real[Nx*i + j] /= norm;
            psi_imag[Nx*i + j] = 0.f;
            test += psi_real[Nx*i + j] * psi_real[Nx*i + j] * dx*dy;
        }
    }
    cout << "Initialized wavefunction: norm = " << std::scientific << test << std::fixed << endl;

    // test = 0;
    // for (int i = 0; i < Ny; ++i) {
    //     for (int j = 0; j < Nx; ++j) {
    //         float x = (i - Ny/2) * dx;
    //         float y = (j - Nx/2) * dx;
    //         test += psi_real[Nx*i + j] * psi_real[Nx*i + j] * dx*dy;
    //     }
    // }
    // cout << "Serial result:" << endl;
    // cout << "Initialized wavefunction: norm = " << std::scientific << test << std::fixed << endl;
}

void HilbertSpace::LeapFrogStepGPU() {
    // Advance the imaginary part before the real one
    ker_laplace(Nx, Ny, dx*mx, dy*my, dt/2, o_psi_real, o_psi_imag);
    ker_harm(Nx, Ny, dx, dy, k_xx, k_yy, k_xy, -dt, o_psi_real, o_psi_imag);

    // Advance the real part
    ker_laplace(Nx, Ny, dx*mx, dy*my, -dt/2, o_psi_imag, o_psi_real);
    ker_harm(Nx, Ny, dx, dy, k_xx, k_yy, k_xy, dt, o_psi_imag, o_psi_real);
}


void HilbertSpace::LeapFrogGPU(int N_steps) {
    for (int i = 0; i < N_steps; ++i) {
        LeapFrogStepGPU();
    }
}

void HilbertSpace::CNGPU(int N_steps) {
    for (int i = 0; i < N_steps; ++i) {
        ApplyCayleyOperator();
    }
}

void HilbertSpace::ApplyInverseHalfEulerStep(memory psi_real, memory psi_imag, int Ntimes) {
    // Set the work to 0
    for (int i = 0; i < Ntimes; ++i) { 
        ker_refresh(Nx, Ny, o_work);

        // Prepare the imaginary part
        ker_laplace(Nx, Ny, dx*mx, dy*my, dt/2, psi_real, o_work);
        ker_harm(Nx, Ny, dx, dy, k_xx, k_yy, k_xy, -dt, psi_real, o_work);

        // Advance the real part
        ker_laplace(Nx, Ny, dx*mx, dy*my, -dt/2, o_psi_imag, psi_real);
        ker_harm(Nx, Ny, dx, dy, k_xx, k_yy, k_xy, dt, o_psi_imag, psi_real);

        // Advance the imaginary part
        ker_sum(Nx, Ny, o_work, psi_imag);
    }
}

void HilbertSpace::ApplyCayleyOperator() {
    // Allocate the memory for the temporaney variable
    memory tmp_psi_real, tmp_psi_imag, work1, work2;
    tmp_psi_real = dev.malloc(Nx*Ny, occa::dtype::get<float>());
    tmp_psi_imag = dev.malloc(Nx*Ny, occa::dtype::get<float>());
    work1 = dev.malloc(Nx*Ny, occa::dtype::get<float>());
    work2 = dev.malloc(Nx*Ny, occa::dtype::get<float>());

    // Apply the direct evolution
    ApplyInverseHalfEulerStep(o_psi_real, o_psi_imag, 1);
    
    // Refresh
    ker_refresh(Nx, Ny, work1);
    ker_refresh(Nx, Ny, work2);

    // Apply the inverse evolution
    for (int i = 0; i < NMAXITER; ++i) {

        // Perform a DeviceToDevice copy (fast)
        tmp_psi_real.copyFrom(o_psi_real);
        tmp_psi_imag.copyFrom(o_psi_real);

        // Apply the hamiltonian
        ApplyInverseHalfEulerStep(tmp_psi_real, tmp_psi_imag, i);

        // Sum
        ker_sum(Nx, Ny, tmp_psi_real, work1);
        ker_sum(Nx, Ny, tmp_psi_imag, work2);
    }

    // Sum
    ker_sum(Nx, Ny, work1, o_psi_real);
    ker_sum(Nx, Ny, work2, o_psi_imag);
}


void HilbertSpace::MeasureCPU(float &X0, float &Y0, float &sigmaX, float &sigmaY, float &norm) {
    X0 = 0;
    Y0 = 0;
    sigmaX = 0;
    sigmaY = 0;
    norm = 0;

    #pragma omp parallel for collapse(2) reduction(+:X0,Y0,sigmaX,sigmaY,norm)
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
            norm += p;
        }
    }
    sigmaX -= X0*X0;
    sigmaY -= Y0*Y0;
}



void HilbertSpace::ConfigureFromCFG(const char * fname) {
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

    dt = 1.0;
    try {
        Nx = cfg.lookup("Nx");
        Ny = cfg.lookup("Ny");
        dx = cfg.lookup("dx");
        dy = cfg.lookup("dy");
        k_xx = cfg.lookup("k_xx");
        k_xy = cfg.lookup("k_xy");
        k_yy = cfg.lookup("k_yy");
        if (!cfg.lookupValue("dt", dt)) {
            cout << "dt" << " not found in the input file." << endl;
            cout << "I assume dt = " << dt << endl;
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
    Setting &root = cfg.getRoot();

    algorithm = ALGORITHM_LF;
    try {
        Setting &Simulation = root["Simulation"];

        mx = Simulation.lookup("mx");
        my = Simulation.lookup("my");

        if (!root.lookupValue("algorithm", algorithm)) {
            cout << "algorithm keyword not found" << endl;
            cout << "assuming " << algorithm << endl;
        }
    } catch (const SettingNotFoundException &e) {
        cerr << "Error, segging " << e.getPath() << " not found." << endl;
        cerr << e.what() << endl;
        throw;
    } catch (const SettingTypeException &e) {
        cerr << "Error, wrong type for setting : " << e.getPath() << endl;
        cerr << e.what() << endl;
        throw;
    }
}

void HilbertSpace::GetEvolveFunctionGPU(int Nsteps) {
    if (algorithm == ALGORITHM_LF) {
        LeapFrogGPU(Nsteps);
    } else if (algorithm == ALGORITHM_CN) {
        CNGPU(Nsteps);
    } else {
        cerr << "Error, I do not understand the algorithm you chosed" << endl;
        cerr << "Algorithm: " << algorithm << endl;
        throw SettingException("wrong algorithm");
    }
}