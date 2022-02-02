/*
Passing variables / arrays between cython and cpp
Example from 
http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html

Adapted to include passing of multidimensional arrays

*/

#include "Rectangle.h"

using namespace shapes;
using namespace arma;

stdvec2d mat_to_std_vec(arma::mat &A) {
    stdvec2d V(A.n_rows);
    for (size_t i = 0; i < A.n_rows; ++i) {
        V[i] = arma::conv_to< stdvec >::from(A.row(i));
    };
    return V;
}

mat std_vec_to_mat(stdvec2d A) {
    int nRow = A.size();
    int nCol = A[0].size();
    arma::mat V(nRow, nCol);
    //arma::Mat V(&A.front(), nRow, nCol);
    for (int i=0; i<nRow; i++){
        for (int j=0; j<nCol; j++)      
                V(i,j)=A[i][j];
    }
    return V;
}

Rectangle::Rectangle(int X0, int Y0, int X1, int Y1)
{
    x0 = X0;
    y0 = Y0;
    x1 = X1;
    y1 = Y1;
}

Rectangle::~Rectangle()
{
}

int Rectangle::getLength()
{
    return (x1 - x0);
}

int Rectangle::getHeight()
{
    return (y1 - y0);
}

int Rectangle::getArea()
{
    return (x1 - x0) * (y1 - y0);
}

void Rectangle::move(int dx, int dy)
{
    x0 += dx;
    y0 += dy;
    x1 += dx;
    y1 += dy;
}

/*
Inputting a 1D vectoror list and returning its sum
*/

double Rectangle::sum_vec(std::vector<double> sv)
{

double tot=0;

int svs = sv.size();
std::cout << "vector length " << svs << std::endl;

for (int ii=0; ii<svs; ii++)
{
        tot = tot + sv.at(ii);
}
        return tot;

}

/*
Inputting a 2D vector or list and returning its sum
*/

double Rectangle::sum_mat(std::vector< std::vector<double> > sv)
{

double tot=0;

int svrows = sv.size();
int svcols = sv[0].size();
std::cout << "vector length " << svrows << " , " << svcols << std::endl;

for (int ii=0; ii<svrows; ii++)
{
        for (int jj=0; jj<svcols; jj++)
        {
                tot = tot + sv.at(ii).at(jj);
        }
}        
return tot;

}


/*
Passing a 2D vector by reference or list and returning its sum
*/

double Rectangle::sum_mat_ref(const stdvec2d & sv)
{

double tot=0;

int svrows = sv.size();
int svcols = sv[0].size();
std::cout << "vector length " << svrows << " , " << svcols << std::endl;

for (int ii=0; ii<svrows; ii++)
{
        for (int jj=0; jj<svcols; jj++)
        {
                tot = tot + sv.at(ii).at(jj);
        }
}        
return tot;

}


/*
Inputting a 2D vector, performing a simple operation and returning a new 2D vector
*/
std::vector< std::vector<double> > Rectangle::ret_mat(std::vector< std::vector<double> > sv)
{

int svrows = sv.size();
int svcols = sv[0].size();

std::vector< std::vector<double> > tot;
tot.resize(svrows, std::vector<double> (svcols, -1));


std::cout << "vector length " << svrows << " , " << svcols << std::endl;

for (int ii=0; ii<svrows; ii++)
{
        for (int jj=0; jj<svcols; jj++)
        {
                tot.at(ii).at(jj) = (2*sv.at(ii).at(jj));
        }
}        
return tot;
}


/*
Inputting a 2D vector, performing matrix multiplication and return a 2D vector
*/
stdvec2d Rectangle::multiply_mat(stdvec2d & vec_a, stdvec2d & vec_b)
{

arma::mat A = std_vec_to_mat(vec_a);
arma::mat B = std_vec_to_mat(vec_b);
arma::mat C = A*B;
stdvec2d vec_result= mat_to_std_vec(C);
return vec_result;
}

