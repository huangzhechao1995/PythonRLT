src/main.cpp#include <pybind11/pybind11.h>
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


    arma::vec prediction = pythonFriend.pythonCallWithGivenTrainTestData(mat_trainx, vec_trainy, mat_testx, vec_testy, ntrees);


    double* prediction_mem = prediction.memptr();
    auto result = py::array_t<double>(buf_trainy.size);
    py::buffer_info buf_result = result.request();
    buf_result.ptr = (double*)prediction.memptr();

    std::cout << "number of elements in prediction: "<< prediction.n_elem<< std::endl;

    return result;
}


List pythonRegWithGivenXYReturnList(py::array_t<double>& trainx, py::array_t<double>& trainy, py::array_t<double>& testx, py::array_t<double>& testy, int ntrees){
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

    List result = pythonFriend.pythonCallWithGivenTrainTestDataReturnList(mat_trainx, vec_trainy, mat_testx, vec_testy, ntrees);

    return result;
}

py::array_t<double> List::getPrediction(){
    double* prediction_mem = Prediction.memptr();
    auto result = py::array_t<double>(Prediction.size());
    py::buffer_info buf_result = result.request();
    buf_result.ptr = (double*)Prediction.memptr();
    return result;
}

py::array_t<double> List::getOOBPrediction(){
    double* prediction_mem = OOBPrediction.memptr();
    auto result = py::array_t<double>(OOBPrediction.size());
    py::buffer_info buf_result = result.request();
    buf_result.ptr = (double*)OOBPrediction.memptr();
    return result;
}

py::array_t<double> List::getVarImp(){
    double* prediction_mem = VarImp.memptr();
    auto result = py::array_t<double>(VarImp.size());
    py::buffer_info buf_result = result.request();
    buf_result.ptr = (double*)VarImp.memptr();
    return result;
}


PYBIND11_MODULE(cmake_example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           
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

    m.def("pythonRegWithGivenXYReturnList", &pythonRegWithGivenXYReturnList, R"pbdoc(
        Pass in trainX, testX, trainY, testY
    )pbdoc");

    py::class_<List> List(m, "List");
    List.def(py::init<>())
        .def("getOOBPrediction", &List::getOOBPrediction)
        .def("getPrediction", &List::getPrediction)
        .def("getVarImp", &List::getVarImp);
        

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

