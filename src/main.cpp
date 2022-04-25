#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "regForest.h"
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
int add(int i, int j) {
    return i + j;
}
using namespace arma;

py::array_t<double> pythonRegWithGivenXY(py::array_t<double>& trainx, py::array_t<double>& trainy, py::array_t<double>& testx, py::array_t<double>& testy, int ntrees){
    py::buffer_info buf_trainx = trainx.request();
    py::buffer_info buf_trainy = trainy.request();
    py::buffer_info buf_testx = testx.request();
    py::buffer_info buf_testy = testy.request();
    if (buf_trainx.ndim != 2 || buf_testx.ndim != 2)
    {
        throw std::runtime_error("numpy.ndarray dims for X  must be 2!");
    }
    if (buf_trainy.ndim != 1 || buf_testy.ndim != 1)
    {
        throw std::runtime_error("numpy.ndarray dims for y must be 2!");
    }
    if ((buf_trainx.shape[0] != buf_trainy.shape[0]))
    {
        throw std::runtime_error("length of train X and train Y must be match!");
    }
    if ((buf_testx.shape[0] != buf_testy.shape[0]))
    {
        throw std::runtime_error("length of testX and testY must be match!");
    }
    double* ptr_trainx = (double*)buf_trainx.ptr;
    double* ptr_trainy = (double*)buf_trainy.ptr;

    double* ptr_testx = (double*)buf_testx.ptr;
    double* ptr_testy = (double*)buf_testy.ptr;

    arma::mat mat_trainx = arma::mat(ptr_trainx, buf_trainx.shape[0], buf_trainx.shape[1],  true, false);
    arma::vec vec_trainy = arma::vec(ptr_trainy, buf_trainy.shape[0], true, false);

    arma::mat mat_testx = arma::mat(ptr_testx, buf_testx.shape[0], buf_testx.shape[1] , true, false);
    arma::vec vec_testy = arma::vec(ptr_testy, buf_testy.shape[0], true, false);

    pythonInterfaceClass pythonFriend = pythonInterfaceClass();
    auto result = py::array_t<double>(buf_trainx.size);
    arma::vec prediction = pythonFriend.pythonCallWithGivenTrainTestData(mat_trainx, vec_trainy, mat_testx, vec_testy, ntrees);
    std::cout << "number of elements in prediction: "<< prediction.n_elem<< std::endl;
    return result;
}


PYBIND11_MODULE(cmake_example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.def("pythonInterface", [](int trainn, int testn, int p, int ntrees) { 
		    pythonInterfaceClass pythonFriend = pythonInterfaceClass();
		    int result=pythonFriend.pythonCallWithRandomData(trainn, testn, p, ntrees);
		    return result;
		    }, R"pbdoc(
        Call RLT

        Some other explanation about the pythonInterface function.
    )pbdoc");

    m.def("pythonRegWithGivenXY", &pythonRegWithGivenXY, R"pbdoc(
        Pass in trainX, testX, trainY, testY
    )pbdoc");
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
