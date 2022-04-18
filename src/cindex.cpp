//  **********************************
//  Reinforcement Learning Trees (RLT)
//  C-index
//  **********************************

// my header file
# include "RLT.h"
# include "Trees//Trees.h"
# include "Utility/Utility.h"

using namespace arma;

// [[Rcpp::export()]]
double cindex_d(arma::vec& Y,
              arma::uvec& Censor,
              arma::vec& pred)
{

  std::cout << "-- calculate cindex (int Y) " << std::endl;
  size_t P = 0;
  double C = 0;
  
  for (size_t i = 0; i < Y.n_elem; i++){
      for (size_t j = 0; j < i; j ++)
      {
        if ( ( Y(i) > Y(j) and Censor(j) == 0 ) or ( Y(i) < Y(j) and Censor(i) == 0 ) )
        {
          continue;
        }
        
        if ( Y(i) == Y(j) and Censor(i) == 0 and Censor(j) == 0 )
        {
          continue;
        }
        
        P++;
        
        if ( Y(i) > Y(j) )
        {
          if ( pred(i) < pred(j) )
          {
            C++;
          }
          
          if ( pred(i) == pred(j) )
          {
            C += 0.5;
          }
          
        }else if( Y(i) < Y(j) ){
          if ( pred(j) < pred(i) )
          {
            C++;
          }
          
          if ( pred(i) == pred(j) )
          {
            C += 0.5;
          }
          
        }else{
          
          if ( Censor(i) == 1 and Censor(j) == 1 )
          {
            if ( pred(i) == pred(j) )
            {
              C++;
            }else{
              C += 0.5;
            }
              
          }else if ( ( Censor(i) == 1 and pred(i) > pred(j) ) or ( Censor(j) == 1 and pred(j) > pred(i) ) )
          {
            C++;
          }else if ( pred(i) == pred(j) )
          {
            C += 0.5;
          }
        }
      }}
  
  return C/P;
}

double cindex_i(arma::uvec& Y,
              arma::uvec& Censor,
              arma::vec& pred)
{
  
  std::cout << "-- calculate cindex (int Y) " << std::endl;
  size_t P = 0;
  double C = 0;
  
  for (size_t i = 0; i < Y.n_elem; i++){
    for (size_t j = 0; j < i; j ++)
    {
      if ( ( Y(i) > Y(j) and Censor(j) == 0 ) or ( Y(i) < Y(j) and Censor(i) == 0 ) )
      {
        continue;
      }
      
      if ( Y(i) == Y(j) and Censor(i) == 0 and Censor(j) == 0 )
      {
        continue;
      }
      
      P++;
      
      if ( Y(i) > Y(j) )
      {
        if ( pred(i) < pred(j) )
        {
          C++;
        }
        
        if ( pred(i) == pred(j) )
        {
          C += 0.5;
        }
        
      }else if( Y(i) < Y(j) ){
        if ( pred(j) < pred(i) )
        {
          C++;
        }
        
        if ( pred(i) == pred(j) )
        {
          C += 0.5;
        }
        
      }else{
        
        if ( Censor(i) == 1 and Censor(j) == 1 )
        {
          if ( pred(i) == pred(j) )
          {
            C++;
          }else{
            C += 0.5;
          }
          
        }else if ( ( Censor(i) == 1 and pred(i) > pred(j) ) or ( Censor(j) == 1 and pred(j) > pred(i) ) )
        {
          C++;
        }else if ( pred(i) == pred(j) )
        {
          C += 0.5;
        }
      }
    }}
  
  return C/P;
  
}




// [[Rcpp::export()]]
arma::umat ARMA_EMPTY_UMAT()
{
  arma::umat temp;
  return temp;
}

// [[Rcpp::export()]]
arma::vec ARMA_EMPTY_VEC()
{
  arma::vec temp;
  return temp;
}










