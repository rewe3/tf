#ifndef MATRIX_DATA_H_
#define MATRIX_DATA_H_

#include <string>
#include <vector>

#include <armadillo>


struct Sparse3dTensor {
  // number of non-zero entries
  int n_nz;
  
  // size in each dim
  int n_u;
  int n_v;
  int n_t;

  // indices
  std::vector<unsigned int> uindex;
  std::vector<unsigned int> vindex;
  std::vector<unsigned int> tindex;
  
  // for iterating through all uv in R
  int n_uv;
  int i;
  arma::frowvec nextWordBag();
  bool hasNext();
  void resetIt();
};


// Parse the split tensor data
struct TensorData {
  int offset;
  Sparse3dTensor R;

  static struct TensorData parse(std::string path);

  static arma::sp_fmat parseTensor(std::ifstream& f);
};

#endif  // MATRIX_DATA_H_

