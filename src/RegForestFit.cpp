//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "Utility/Utility.h"
# include "regForest.h"
# include <iostream>
# include "Trees/Definition.h"
# include <string>
# include "cindex.cpp"

using namespace arma;

// [[Rcpp::export()]]
List RegForestUniFit(arma::mat& X,
          					 arma::vec& Y,
          					 arma::uvec& Ncat,
          					 PARAM_GLOBAL& Param,
          					 PARAM_RLT& Param_RLT,
          					 arma::vec& obsweight,
          					 arma::vec& varweight,
          					 int usecores,
          					 int verbose,
          					 arma::umat& ObsTrack)
{
  std::cout << "/// THIS IS A DEBUG MODE OF RLT REGRESSION ///" << std::endl;

  std::cout<<Param.N <<Param.P << std::endl;

  // check number of cores
  usecores = checkCores(usecores, verbose);

  // readin parameters 
  // param.print();
  // PARAM_GLOBAL Param = param;
  if (verbose) Param.print();
  // PARAM_RLT Param_RLT = RLTparam;
  if (verbose and Param.reinforcement) Param_RLT.print();

  // create data objects  
  RLT_REG_DATA REG_DATA(X, Y, Ncat, obsweight, varweight);
  
  size_t N = REG_DATA.X.n_rows;
  size_t P = REG_DATA.X.n_cols;
  size_t ntrees = Param.ntrees;
  size_t seed = Param.seed;
  int obs_track = Param.obs_track;

  int importance = Param.importance;

  // initiate forest
  arma::field<arma::uvec> NodeType(ntrees);
  arma::field<arma::uvec> SplitVar(ntrees);
  arma::field<arma::vec> SplitValue(ntrees);
  arma::field<arma::uvec> LeftNode(ntrees);
  arma::field<arma::uvec> RightNode(ntrees);
  arma::field<arma::vec> NodeSize(ntrees);
  arma::field<arma::vec> NodeAve(ntrees);

  Reg_Uni_Forest_Class REG_FOREST(NodeType, SplitVar, SplitValue, LeftNode, RightNode, NodeSize, NodeAve);
  
  // other objects

  // VarImp
  vec VarImp;
  
  if (importance)
    VarImp.zeros(P);
  
  // prediction
  vec Prediction;
  vec OOBPrediction;
  
  // initiate obs id and var id
  uvec obs_id = linspace<uvec>(0, N-1, N);
  uvec var_id = linspace<uvec>(0, P-1, P);
  
  //start to fit the model
  Reg_Uni_Forest_Build((const RLT_REG_DATA&) REG_DATA,
                       REG_FOREST,
                       (const PARAM_GLOBAL&) Param,
                       (const PARAM_RLT&) Param_RLT,
                       obs_id,
                       var_id,
                       ObsTrack,
                       Prediction,
                       OOBPrediction,
                       VarImp,
                       seed,
                       usecores,
                       verbose);

  List ReturnList;
  
  Forest Forest_R;
  
  Forest_R.NodeType = NodeType;
  Forest_R.SplitVar = SplitVar;
  Forest_R.SplitValue = SplitValue;
  Forest_R.LeftNode = LeftNode;
  Forest_R.RightNode = RightNode;
  Forest_R.NodeSize = NodeSize;    
  Forest_R.NodeAve = NodeAve;
  
  ReturnList.FittedForest = Forest_R;
  
  if (obs_track) ReturnList.ObsTrack = ObsTrack;
  if (importance) ReturnList.VarImp = VarImp;
  
  ReturnList.Prediction = Prediction;
  ReturnList.OOBPrediction = OOBPrediction;

  return ReturnList;
}


int pythonInterface(int trainn, int testn, int p, int ntrees){
  // from 
  // int trainn = 500;
  // int testn = 1000;
  int n = trainn + testn;
  std::cout<<n<<std::endl;
  // int p = 100;
  //int X1 = matrix(rnorm(n*p/2), n, p/2)
  // int ntrees = 200;
  int ncores = 10;
  int nmin = 20;
  int mtry = p;
  double sampleprob = 0.85;
  std::string rule = "best";
  int nsplit = 0;
  int importance = 1; 

  arma::mat dummy_X = arma::mat(n, p, fill::randu);
  arma::vec dummy_Y = arma::vec(n, fill::randu);
  arma::uvec dummy_Ncat = arma::uvec(p, fill::zeros);
  return importance;
  PARAM_GLOBAL dummy_param = PARAM_GLOBAL( n,
                            p,
                            ntrees,
                            mtry,
                            nmin,
                            0.0, //default
                            1,
                            1,
                            nsplit,
                            true,
                            sampleprob,
                            false,
                            false,
                            1,
                            1,
                            false,
                            false,
                            312,
                            false);
  std::cout<< "dummy_param.N: " <<dummy_param.N<<std::endl;
  std::cout<< "dummy_param.P: " <<dummy_param.P<<std::endl;
    
  PARAM_RLT dummy_RLTparam = PARAM_RLT(
    1,
    0.75,
    0.33,
    1,
    1 ,
    1
    );
  
  std::cout<< "dummy_RLTparam.embed_ntrees" <<dummy_RLTparam.embed_ntrees<<std::endl;

  arma::vec dummy_obsweight;
  arma::vec dummy_varweight;
  int dummy_usecores;
  int dummy_verbose;
  arma::umat dummy_ObsTrack;
  List fit_result = RegForestUniFit(dummy_X, dummy_Y, dummy_Ncat, dummy_param, dummy_RLTparam, dummy_obsweight, dummy_varweight, dummy_usecores, dummy_verbose, dummy_ObsTrack);
  
  return dummy_param.N;
}

int pythonInterfaceClass::pythonCall(int trainn, int testn, int p, int ntrees){
  int result = pythonInterface(trainn, testn, p, ntrees);
  return result;
}

//int main(){
//    pythonInterfaceClass interf = pythonInterfaceClass();
//    interf.pythonCall(1000, 500, 100, 200);
//}

