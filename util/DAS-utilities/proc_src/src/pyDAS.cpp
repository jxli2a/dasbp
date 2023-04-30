#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <omp.h>
#include "filters.h"

using namespace std;
namespace py = pybind11;


// Exposing function
py::array_t<float> lfilter(py::array_t<float> in_arr, float flo, int npho, float fhi, int nphi, int phase, int nthreads) {
    // // GIL release  
    // py::gil_scoped_release release;
    // // Acquire GIL before calling Python code 
    // py::gil_scoped_acquire acquire;


    auto buf_in = in_arr.request();
    if (buf_in.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two!");
    
    // Getting size of the input array
    int nt = in_arr.shape(1);
    int nch = in_arr.shape(0);
    int ntpad = nt+2;

    // Allocating output array
    py::array_t<float> out_arr = py::array_t<float>(buf_in.size);
    auto buf_out = out_arr.request();


    // Checking if input cutoff frequencies are correct
    if (flo < 0.0001 && fhi > 0.4999) {
        throw std::runtime_error("Incorrect cutoff frequencies!");
    }

    // Filtering order
    if (npho < 1) npho = 1;
    if (nphi < 1) nphi = 1;

    // Get pointers to data arrays
    float * data_p = (float *) buf_in.ptr;
    float * data_o = (float *) buf_out.ptr;

    // Checking number of threads 
    nthreads = std::min(nch, nthreads);

    // Allocating temporary arrays
    auto ** data = new float*[nthreads];
    auto ** newdata = new float*[nthreads];
    auto ** tempdata = new float*[nthreads];
    for (int ithread = 0; ithread < nthreads; ithread++){
        data[ithread] = new float[ntpad];
        newdata[ithread] = new float[ntpad];
        tempdata[ithread] = new float[ntpad];
    }

    #pragma omp parallel for schedule(dynamic,1) num_threads(nthreads)
    for (long long ich = 0; ich < nch; ich++){
        int ithread = omp_get_thread_num();

        // Zeroing out temporary arrays
        std::memset(data[ithread], 0, ntpad*sizeof(float));
        std::memset(newdata[ithread], 0, ntpad*sizeof(float));
        std::memset(tempdata[ithread], 0, ntpad*sizeof(float));

        // Copying input data into temporary arrray
        std::memcpy(data[ithread]+2, data_p+ich*nt, nt*sizeof(float));
  
        // Applying highcut filter
        if (flo > 0.0001){
            lowcut(flo, npho, phase, ntpad, data[ithread], newdata[ithread], tempdata[ithread]);
        }

        // Applying highcut filter
        if (fhi < 0.4999){
            highcut(fhi, nphi, phase, ntpad, data[ithread], newdata[ithread], tempdata[ithread]);
        }

        // Copying result to output array
        std::memcpy(data_o+ich*nt, data[ithread]+2, nt*sizeof(float));
   
    }

    // Deallocating temporary memory
    for (int ithread = 0; ithread < nthreads; ithread++){
        delete data[ithread];
        delete newdata[ithread];
        delete tempdata[ithread];
    }
    delete [] data;
    delete [] newdata;
    delete [] tempdata;

    // Reshape output array
    out_arr.resize({nch, nt});
        
    return out_arr;

}


// Deciding what to expose in the library python can import
PYBIND11_MODULE(pyDAS, m) {
    m.doc() = "pybind11 module for DAS data processing";
    m.def("lfilter", &lfilter, "Apply linear filter (bandpass, high-pass, low-pass)");
}