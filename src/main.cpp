#include <iostream>
#include <string>
#include <libconfig.h++>
#include <fstream>
#include <sys/time.h>

#include "EvolveWFC.hpp"
using namespace std;
using namespace libconfig;

void LoadInfoSym(const char * fname, int &Nt, int &Nsteps, float &mx, float &my, string &save);
void WriteToFile(const char * fname, int Nx, int Ny, const float * psi);

int main(int argc, char * argv[]) {
    // Check if there is an input file
    if (argc < 2) {
        cerr << "Error, please provide an input file." << endl;
        exit(EXIT_FAILURE);
    }

    // Create a new hilbert space
    HilbertSpace HS = HilbertSpace(argv[1]);

    struct timeval startTime, endTime;

    int Nt, Nsteps;
    string save_fname;
    string real_name;
    LoadInfoSym(argv[1], Nt, Nsteps, HS.mx, HS.my, save_fname);

    float X0, Y0, sigmaX, sigmaY, sigmaXY, norm;

    // Run
    HS.CopyToGPU();
    for (int i = 0; i < Nt; i++) {
        cout << " ======  STEP " << i << " =======" << endl;
        // Perform the simulation
        gettimeofday(&startTime, NULL);
        HS.EvolveFunctionGPU(Nsteps);

        // Copy back in the CPU
        HS.CopyFromGPU();
        gettimeofday(&endTime, NULL);

        float delta = (endTime.tv_sec - startTime.tv_sec) * 1000.f + (endTime.tv_usec - startTime.tv_usec) / 1000.f;
        cout << "Time spent for a single computation is " << delta << " ms" << endl;

        // Now save the data
        real_name = save_fname + "_real_" + to_string(i) + ".dat";
        cout << endl << "Writing wavefunction on file " << real_name << endl;
        WriteToFile(real_name.c_str(), HS.Nx, HS.Ny, HS.psi_real);
        real_name = save_fname + "_imag_" + to_string(i) + ".dat";
        cout << "Writing wavefunction on file " << real_name << endl << endl;
        WriteToFile(real_name.c_str(), HS.Nx, HS.Ny, HS.psi_imag);

        // Compute the observables
        HS.MeasureCPU(X0, Y0, sigmaX, sigmaY, norm);

        // Print the observables on stdout
        cout << endl;
        cout << "Time = " << (i+1) * Nsteps * HS.dt << endl;
        cout << "Averages:" << endl;
        cout << std::scientific;
        cout << "Normalization = " << norm << endl;
        cout << "X0 = " << X0 << endl;
        cout << "Y0 = " << Y0 << endl;
        cout << "Sigma2X = " << sigmaX << endl;
        cout << "Sigma2Y = " << sigmaY << endl;
        cout << std::fixed ;
        cout << endl;
    }
    return EXIT_SUCCESS;
}

void LoadInfoSym(const char * fname, int &Nt, int &Nsteps, float &mx, float &my, string &save) {
    Config cfg;
    cfg.readFile(fname);

    const Setting & root = cfg.getRoot();
    try {
        const Setting & Simulation = root["Simulation"];

        Nt = Simulation.lookup("Nt");
        Nsteps = Simulation.lookup("Nsteps");
        mx = Simulation.lookup("mx");
        my = Simulation.lookup("my");
        if (!Simulation.lookupValue("save_dest", save)) {
            cerr << "Error, the 'save_dest' key not found into the 'Simulation' environment" << endl;
            throw SettingNotFoundException("save_dest");
        } 
    } catch (const SettingNotFoundException &e) {
        cerr << "Error, setting " << e.getPath() << " not found." << endl;
        cerr << e.what() << endl;
        throw;
    } catch (const SettingTypeException &e) {
        cerr << "Error, setting " << e.getPath() << " is of the wrong type" << endl;
        cerr << e.what() << endl;
        throw;
    }
}

void WriteToFile(const char * fname, int Nx, int Ny, const float * psi) {
    ofstream file(fname);

    if (!file.is_open()) {
        cerr << "Error while opening file " << fname << " to write." << endl;
        throw FileIOException();
    }

    // Write numbers in scientific notation
    file << std::scientific;
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            file << psi[Nx*i + j] << "\t";
        }
        file << endl;
    }
    file.close();
}