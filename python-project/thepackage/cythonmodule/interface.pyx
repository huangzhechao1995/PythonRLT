# distutils: language = c++

# cdef extern from "regForest.h":
#   cpdef int pythonCall(int trainn, int testn, int p, int ntrees)

cdef extern from "regForest.h":
  cdef cppclass pythonInterfaceClass:
    int pythonCall(int trainn, int testn, int p, int ntrees)

cdef class PyPythonInterfaceClass:
  def __dealloc__(self):
        del self.thisptr
  def pythonCall(self, int trainn, int testn, int p, int ntrees):
        return self.thisptr.pythonCall(trainn,testn,p,ntrees)
 