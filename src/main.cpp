#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "regForest.h"
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
int add(int i, int j)
{
    return i + j;
}
using namespace arma;

List pythonRegWithGivenXYReturnList(py::array_t<double> &trainx, py::array_t<double> &trainy, py::array_t<double> &testx, py::array_t<double> &testy, int ntrees)
{
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
    double *ptr_trainx = (double *)buf_trainx.ptr;
    double *ptr_trainy = (double *)buf_trainy.ptr;

    double *ptr_testx = (double *)buf_testx.ptr;
    double *ptr_testy = (double *)buf_testy.ptr;

    arma::mat mat_trainx = arma::mat(ptr_trainx, buf_trainx.shape[0], buf_trainx.shape[1], true, false);
    arma::vec vec_trainy = arma::vec(ptr_trainy, buf_trainy.shape[0], true, false);

    arma::mat mat_testx = arma::mat(ptr_testx, buf_testx.shape[0], buf_testx.shape[1], true, false);
    arma::vec vec_testy = arma::vec(ptr_testy, buf_testy.shape[0], true, false);

    // verification
    std::cout << "train y mean in C++" << arma::sum(vec_trainy) << std::endl;
    std::cout << "test y mean in C++" << arma::sum(vec_testy) << std::endl;

    pythonInterfaceClass pythonFriend = pythonInterfaceClass();

    List result = pythonFriend.pythonCallWithGivenTrainTestDataReturnList(mat_trainx, vec_trainy, mat_testx, vec_testy, ntrees);

    return result;
}

py::array_t<double> List::getPrediction()
{
    auto result = py::array_t<double>(Prediction.size());
    for (int i; i<Prediction.size(); ++i) {
        result.mutable_at(i) = Prediction[i];
    }
    std::cout << "List::getPrediction = " << Prediction << std::endl;
    return result;
}

py::array_t<double> List::getOOBPrediction()
{
    auto result = py::array_t<double>(OOBPrediction.size());
    for (int i; i<OOBPrediction.size(); ++i) {
        result.mutable_at(i) = OOBPrediction[i];
    }
    std::cout << "List::getOOBPrediction = " << OOBPrediction << std::endl;
    return result;
}
py::array_t<double> List::getTestPrediction()
{
    auto result = py::array_t<double>(TestPrediction.size());
    for (int i; i<TestPrediction.size(); ++i) {
        result.mutable_at(i) = TestPrediction[i];
    }
    std::cout << "List::getTestPrediction = " << TestPrediction << std::endl;
    return result;
}
py::array_t<double> List::getVarImp()
{
    auto result = py::array_t<double>(VarImp.size());
    for (int i; i<VarImp.size(); ++i) {
        result.mutable_at(i) = VarImp[i];
    }
    std::cout << "List::getVarImp = " << VarImp << std::endl;
    return result;
}

// py::array_t<double> List::getTestPrediction()
// {
//     double *prediction_mem = TestPrediction.memptr();
//     auto result = py::array_t<double>(TestPrediction.size());
//     py::buffer_info buf_result = result.request();
//     buf_result.ptr = (double *)TestPrediction.memptr();
//     return result;
// }
List pythonRegPrediction(py::array_t<double> &testx, List fit)
{
    std::cout << "start to run prediction" << std::endl;
    py::buffer_info buf_testx = testx.request();
    double *ptr_testx = (double *)buf_testx.ptr;
    arma::mat mat_testx = arma::mat(ptr_testx, buf_testx.shape[0], buf_testx.shape[1], true, false);
    pythonInterfaceClass pythonFriend = pythonInterfaceClass();

    // pythonFriend.pythonCallPredictOnTestData(mat_testx, fit);
    arma::vec prediction(5, fill::value(123.0));
    // prediction = pythonFriend.pythonCallPredictOnTestData(mat_testx, fit);

    std::cout << "prediction result inside C++ is" << prediction << prediction.memptr() << std::endl;
    // // cast result
    // double *prediction_mem = prediction.memptr();
    // auto result = py::array_t<double>(prediction.size());
    // py::buffer_info buf_result = result.request();
    // py::array_t<double> final_result(buf_result);

    List ReturnList;
    ReturnList.TestPrediction = prediction;
    return ReturnList;
}

PYBIND11_MODULE(pythonrlt, m)
{
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: pythonrlt

        .. autosummary::
           :toctree: _generate

           
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def(
        "subtract", [](int i, int j)
        { return i - j; },
        R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.def(
        "pythonInterface", [](int trainn, int testn, int p, int ntrees)
        {
            pythonInterfaceClass pythonFriend = pythonInterfaceClass();
            int result = pythonFriend.pythonCallWithRandomData(trainn, testn, p, ntrees);
            return result; },
        R"pbdoc(
        Call RLT

        Some other explanation about the pythonInterface function.
    )pbdoc");

    m.def("pythonRegWithGivenXYReturnList", &pythonRegWithGivenXYReturnList, R"pbdoc(
        Pass in trainX, testX, trainY, testY
    )pbdoc");

    m.def("pythonRegPrediction", &pythonRegPrediction, R"pbdoc(
        Pass in testX, fit
    )pbdoc");

    // py::class_<pythonInterface> List(m, "pythonInterface");

    py::class_<List> List(m, "List");
    List.def(py::init<>())
        .def("getOOBPrediction", &List::getOOBPrediction)
        .def("getPrediction", &List::getPrediction)
        .def("getVarImp", &List::getVarImp)
        .def("getTestPrediction", &List::getTestPrediction);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
