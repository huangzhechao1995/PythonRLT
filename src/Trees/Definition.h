//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file

#include <iostream>
#include <armadillo>
using namespace arma;

#define SurvWeightTH 1e-20

#ifndef RLT_DEFINITION
#define RLT_DEFINITION

class PARAM_GLOBAL
{
public:
  size_t N;
  size_t P;
  size_t ntrees;
  size_t mtry;
  size_t nmin;
  double alpha;
  int split_gen;
  int split_rule;
  int nsplit;
  bool replacement;
  double resample_prob;
  bool useobsweight;
  bool usevarweight;
  int varweighttype;
  int importance;
  bool reinforcement;
  bool obs_track;
  size_t seed;
  bool failcount;

  PARAM_GLOBAL(size_t N_input,
               size_t P_input,
               size_t ntrees_input,
               size_t mtry_input,
               size_t nmin_input,
               double alpha_input,
               int split_gen_input,
               int split_rule_input,
               int nsplit_input,
               bool replacement_input,
               double resample_prob_input,
               bool useobsweight_input,
               bool usevarweight_input,
               int varweighttype_input,
               int importance_input,
               bool reinforcement_input,
               bool obs_track_input,
               size_t seed_input,
               bool failcount_input)
  {
    N = N_input;
    P = P_input;
    ntrees = ntrees_input;
    mtry = mtry_input;
    nmin = nmin_input;
    alpha = alpha_input;
    split_gen = split_gen_input;
    split_rule = split_rule_input;
    nsplit = nsplit_input;
    replacement = replacement_input;
    resample_prob = resample_prob_input;
    useobsweight = useobsweight_input;
    usevarweight = usevarweight_input;
    varweighttype = varweighttype_input;
    importance = importance_input;
    reinforcement = reinforcement_input;
    obs_track = obs_track_input;
    seed = seed_input;

    std::cout << "before initialization dummy_param.failcount" << failcount << std::endl;
    failcount = failcount_input;
    std::cout << "during initialization dummy_param.failcount" << failcount << std::endl;
  }

  PARAM_GLOBAL(const PARAM_GLOBAL &Input)
  {
    N = Input.N;
    P = Input.P;
    ntrees = Input.ntrees;
    mtry = Input.mtry;
    nmin = Input.nmin;
    alpha = Input.alpha;
    split_gen = Input.split_gen;
    split_rule = Input.split_rule;
    nsplit = Input.nsplit;
    replacement = Input.replacement;
    resample_prob = Input.resample_prob;
    useobsweight = Input.useobsweight;
    usevarweight = Input.usevarweight;
    importance = Input.importance;
    reinforcement = Input.reinforcement;
    obs_track = Input.obs_track;
    seed = Input.seed;
  }

  void print()
  {
    std::cout << "--- Random Forest Parameters ---" << std::endl;
    std::cout << "            N = " << N << std::endl;
    std::cout << "            P = " << P << std::endl;
    std::cout << "       ntrees = " << ntrees << std::endl;
    std::cout << "         mtry = " << mtry << std::endl;
    std::cout << "         nmin = " << nmin << std::endl;
    std::cout << "        alpha = " << alpha << std::endl;
    std::cout << "    split_gen = " << ((split_gen == 1) ? "Random" : (split_gen == 2) ? "Rank"
                                                                                       : "Best")
              << std::endl;
    if (split_gen < 3)
      std::cout << "   split_rule = " << split_rule << std::endl;
    std::cout << "       nsplit = " << nsplit << std::endl;
    std::cout << "  replacement = " << replacement << std::endl;
    std::cout << "resample prob = " << resample_prob << std::endl;
    std::cout << " useobsweight = " << (useobsweight ? "Yes" : "No") << std::endl;
    std::cout << " usevarweight = " << (usevarweight ? "Yes" : "No") << std::endl;
    std::cout << "   importance = " << (importance ? "Yes" : "No") << std::endl;
    std::cout << "reinforcement = " << (reinforcement ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
  }
};

class PARAM_RLT
{
public:
  size_t embed_ntrees;
  double embed_resample_prob;
  double embed_mtry_prop;
  size_t embed_nmin;
  size_t embed_split_gen;
  size_t embed_nsplit;

  PARAM_RLT(
      size_t embed_ntrees,
      double embed_resample_prob,
      double embed_mtry_prop,
      size_t embed_nmin,
      size_t embed_split_gen,
      size_t embed_nsplit)
  {
    embed_ntrees = embed_ntrees;
    embed_resample_prob = embed_resample_prob;
    embed_mtry_prop = embed_mtry_prop;
    embed_nmin = embed_nmin;
    embed_split_gen = embed_split_gen;
    embed_nsplit = embed_nsplit;
  }

  void print()
  {
    std::cout << "--- Embedded Model Parameters ---" << std::endl;
    std::cout << "        embed_ntrees = " << embed_ntrees << std::endl;
    std::cout << " embed_resample_prob = " << embed_resample_prob << std::endl;
    std::cout << "     embed_mtry_prop = " << embed_mtry_prop << std::endl;
    std::cout << "          embed_nmin = " << embed_nmin << std::endl;
    std::cout << "     embed_split_gen = " << embed_split_gen << std::endl;
    std::cout << "        embed_nsplit = " << embed_nsplit << std::endl;
    std::cout << std::endl;
  }
};

// *************** //
// field functions //
// *************** //

void field_vec_resize(arma::field<arma::vec> &A, size_t size);
void field_vec_resize(arma::field<arma::uvec> &A, size_t size);

// ************ //
//  data class  //
// ************ //

class RLT_REG_DATA
{
public:
  arma::mat &X;
  arma::vec &Y;
  arma::uvec &Ncat;
  arma::vec &obsweight;
  arma::vec &varweight;

  RLT_REG_DATA(arma::mat &X,
               arma::vec &Y,
               arma::uvec &Ncat,
               arma::vec &obsweight,
               arma::vec &varweight) : X(X),
                                       Y(Y),
                                       Ncat(Ncat),
                                       obsweight(obsweight),
                                       varweight(varweight) {}
};

class RLT_SURV_DATA
{
public:
  arma::mat &X;
  arma::uvec &Y;
  arma::uvec &Censor;
  arma::uvec &Ncat;
  size_t NFail;
  arma::vec &obsweight;
  arma::vec &varweight;

  RLT_SURV_DATA(arma::mat &X,
                arma::uvec &Y,
                arma::uvec &Censor,
                arma::uvec &Ncat,
                size_t NFail,
                arma::vec &obsweight,
                arma::vec &varweight) : X(X),
                                        Y(Y),
                                        Censor(Censor),
                                        Ncat(Ncat),
                                        NFail(NFail),
                                        obsweight(obsweight),
                                        varweight(varweight) {}
};

// *********************** //
//  Tree and forest class  //
// *********************** //

class Uni_Tree_Class
{ // univariate split trees
public:
  arma::uvec &NodeType;
  arma::uvec &SplitVar;
  arma::vec &SplitValue;
  arma::uvec &LeftNode;
  arma::uvec &RightNode;
  arma::vec &NodeSize;

  Uni_Tree_Class(arma::uvec &NodeType,
                 arma::uvec &SplitVar,
                 arma::vec &SplitValue,
                 arma::uvec &LeftNode,
                 arma::uvec &RightNode,
                 arma::vec &NodeSize) : NodeType(NodeType),
                                        SplitVar(SplitVar),
                                        SplitValue(SplitValue),
                                        LeftNode(LeftNode),
                                        RightNode(RightNode),
                                        NodeSize(NodeSize) {}

  // find the next left and right nodes
  void find_next_nodes(size_t &NextLeft, size_t &NextRight)
  {
    while (NodeType(NextLeft))
      NextLeft++;
    NodeType(NextLeft) = 1;

    NextRight = NextLeft;
    while (NodeType(NextRight))
      NextRight++;

    // 0: unused, 1: reserved; 2: internal node; 3: terminal node
    NodeType(NextRight) = 1;
  }

  // get tree length
  size_t get_tree_length()
  {
    size_t i = 0;
    while (i < NodeType.n_elem and NodeType(i) != 0)
      i++;
    return ((i < NodeType.n_elem) ? i : NodeType.n_elem);
  }
};

// for regression

class Reg_Uni_Forest_Class
{
public:
  arma::field<arma::uvec> &NodeTypeList;
  arma::field<arma::uvec> &SplitVarList;
  arma::field<arma::vec> &SplitValueList;
  arma::field<arma::uvec> &LeftNodeList;
  arma::field<arma::uvec> &RightNodeList;
  arma::field<arma::vec> &NodeSizeList;
  arma::field<arma::vec> &NodeAveList;

  Reg_Uni_Forest_Class(arma::field<arma::uvec> &NodeTypeList,
                       arma::field<arma::uvec> &SplitVarList,
                       arma::field<arma::vec> &SplitValueList,
                       arma::field<arma::uvec> &LeftNodeList,
                       arma::field<arma::uvec> &RightNodeList,
                       arma::field<arma::vec> &NodeSizeList,
                       arma::field<arma::vec> &NodeAveList) : NodeTypeList(NodeTypeList),
                                                              SplitVarList(SplitVarList),
                                                              SplitValueList(SplitValueList),
                                                              LeftNodeList(LeftNodeList),
                                                              RightNodeList(RightNodeList),
                                                              NodeSizeList(NodeSizeList),
                                                              NodeAveList(NodeAveList) {}
};

class Reg_Uni_Tree_Class : public Uni_Tree_Class
{
public:
  arma::vec &NodeAve;

  Reg_Uni_Tree_Class(arma::uvec &NodeType,
                     arma::uvec &SplitVar,
                     arma::vec &SplitValue,
                     arma::uvec &LeftNode,
                     arma::uvec &RightNode,
                     arma::vec &NodeSize,
                     arma::vec &NodeAve) : Uni_Tree_Class(NodeType,
                                                          SplitVar,
                                                          SplitValue,
                                                          LeftNode,
                                                          RightNode,
                                                          NodeSize),
                                           NodeAve(NodeAve) {}

  // initiate tree
  void initiate(size_t TreeLength)
  {
    if (TreeLength == 0)
      TreeLength = 1;

    NodeType.zeros(TreeLength);

    SplitVar.set_size(TreeLength);
    SplitVar.fill(datum::nan);

    SplitValue.zeros(TreeLength);
    LeftNode.zeros(TreeLength);
    RightNode.zeros(TreeLength);
    NodeSize.zeros(TreeLength);
    NodeAve.zeros(TreeLength);
  }

  // trim tree
  void trim(size_t TreeLength)
  {
    NodeType.resize(TreeLength);
    SplitVar.resize(TreeLength);
    SplitValue.resize(TreeLength);
    LeftNode.resize(TreeLength);
    RightNode.resize(TreeLength);
    NodeSize.resize(TreeLength);
    NodeAve.resize(TreeLength);
  }

  // extend tree
  void extend()
  {
    // tree is not long enough, extend
    size_t OldLength = NodeType.n_elem;
    size_t NewLength = (OldLength * 1.5 > OldLength + 100) ? (size_t)(OldLength * 1.5) : (OldLength + 100);

    NodeType.resize(NewLength);
    NodeType(span(OldLength, NewLength - 1)).zeros();

    SplitVar.resize(NewLength);
    SplitVar(span(OldLength, NewLength - 1)).fill(datum::nan);

    SplitValue.resize(NewLength);
    SplitValue(span(OldLength, NewLength - 1)).zeros();

    LeftNode.resize(NewLength);
    LeftNode(span(OldLength, NewLength - 1)).zeros();

    RightNode.resize(NewLength);
    RightNode(span(OldLength, NewLength - 1)).zeros();

    NodeSize.resize(NewLength);
    NodeSize(span(OldLength, NewLength - 1)).zeros();

    NodeAve.resize(NewLength);
    NodeAve(span(OldLength, NewLength - 1)).zeros();
  }
};

// for survival

class Surv_Uni_Forest_Class
{
public:
  arma::field<arma::uvec> &NodeTypeList;
  arma::field<arma::uvec> &SplitVarList;
  arma::field<arma::vec> &SplitValueList;
  arma::field<arma::uvec> &LeftNodeList;
  arma::field<arma::uvec> &RightNodeList;
  arma::field<arma::vec> &NodeSizeList;
  arma::field<arma::field<arma::vec>> &NodeHazList;

  Surv_Uni_Forest_Class(arma::field<arma::uvec> &NodeTypeList,
                        arma::field<arma::uvec> &SplitVarList,
                        arma::field<arma::vec> &SplitValueList,
                        arma::field<arma::uvec> &LeftNodeList,
                        arma::field<arma::uvec> &RightNodeList,
                        arma::field<arma::vec> &NodeSizeList,
                        arma::field<arma::field<arma::vec>> &NodeHazList) : NodeTypeList(NodeTypeList),
                                                                            SplitVarList(SplitVarList),
                                                                            SplitValueList(SplitValueList),
                                                                            LeftNodeList(LeftNodeList),
                                                                            RightNodeList(RightNodeList),
                                                                            NodeSizeList(NodeSizeList),
                                                                            NodeHazList(NodeHazList) {}
};

class Surv_Uni_Tree_Class : public Uni_Tree_Class
{
public:
  arma::field<arma::vec> &NodeHaz;

  Surv_Uni_Tree_Class(arma::uvec &NodeType,
                      arma::uvec &SplitVar,
                      arma::vec &SplitValue,
                      arma::uvec &LeftNode,
                      arma::uvec &RightNode,
                      arma::vec &NodeSize,
                      arma::field<arma::vec> &NodeHaz) : Uni_Tree_Class(NodeType,
                                                                        SplitVar,
                                                                        SplitValue,
                                                                        LeftNode,
                                                                        RightNode,
                                                                        NodeSize),
                                                         NodeHaz(NodeHaz) {}

  // initiate tree
  void initiate(size_t TreeLength)
  {
    if (TreeLength == 0)
      TreeLength = 1;

    NodeType.zeros(TreeLength);

    SplitVar.set_size(TreeLength);
    SplitVar.fill(datum::nan);

    SplitValue.zeros(TreeLength);
    LeftNode.zeros(TreeLength);
    RightNode.zeros(TreeLength);
    NodeSize.zeros(TreeLength);
    NodeHaz.set_size(TreeLength);
  }

  // trim tree
  void trim(size_t TreeLength)
  {
    NodeType.resize(TreeLength);
    SplitVar.resize(TreeLength);
    SplitValue.resize(TreeLength);
    LeftNode.resize(TreeLength);
    RightNode.resize(TreeLength);
    NodeSize.resize(TreeLength);
    field_vec_resize(NodeHaz, TreeLength);
  }

  // extend tree
  void extend()
  {
    // tree is not long enough, extend
    size_t OldLength = NodeType.n_elem;
    size_t NewLength = (OldLength * 1.5 > OldLength + 100) ? (size_t)(OldLength * 1.5) : (OldLength + 100);

    NodeType.resize(NewLength);
    NodeType(span(OldLength, NewLength - 1)).zeros();

    SplitVar.resize(NewLength);
    SplitVar(span(OldLength, NewLength - 1)).fill(datum::nan);

    SplitValue.resize(NewLength);
    SplitValue(span(OldLength, NewLength - 1)).zeros();

    LeftNode.resize(NewLength);
    LeftNode(span(OldLength, NewLength - 1)).zeros();

    RightNode.resize(NewLength);
    RightNode(span(OldLength, NewLength - 1)).zeros();

    NodeSize.resize(NewLength);
    NodeSize(span(OldLength, NewLength - 1)).zeros();

    field_vec_resize(NodeHaz, NewLength);
  }
};

// **************** //
// class for splits //
// **************** //

class Uni_Split_Class
{ // univariate splits
public:
  size_t var = 0;
  double value = 0;
  double score = -1;

  void print(void)
  {
    std::cout << "Splitting varible is " << var << " value is " << value << " score is " << score << std::endl;
  }
};

// ************************ //
// for categorical variable //
// ************************ //

class Cat_Class
{
public:
  size_t cat = 0;
  size_t count = 0;  // count is used for setting nmin
  double weight = 0; // weight is used for calculation
  double score = 0;  // for sorting

  void print()
  {
    std::cout << "Category is " << cat << " count is " << count << " weight is " << weight << " score is " << score << std::endl;
  }
};

class Reg_Cat_Class : public Cat_Class
{
public:
  double y = 0;

  void calculate_score()
  {
    if (weight > 0)
      score = y / weight;
  }

  void print(void)
  {
    std::cout << "Category is " << cat << " count is " << count << " weight is " << weight << " y sum is " << y << " score is " << score << std::endl;
  }
};

class Surv_Cat_Class : public Cat_Class
{
public:
  arma::uvec FailCount;
  arma::uvec RiskCount;
  size_t nfail;

  void initiate(size_t j, size_t NFail)
  {
    cat = j;
    nfail = 0;
    FailCount.zeros(NFail + 1);
    RiskCount.zeros(NFail + 1);
  }

  void print()
  {
    std::cout << "Category is " << cat << " weight is " << weight << " count is " << count << " data is\n"
              << join_rows(FailCount, RiskCount) << std::endl;
  }

  void print_simple()
  {
    std::cout << "Category is " << cat << " weight is " << weight << " count is " << count << std::endl;
  }
};

#endif
