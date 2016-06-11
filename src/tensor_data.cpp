#include <fstream>

#include <armadillo>

#include "matrix_data.h"

arma::frowvec Sparse3dTensor::nextWordBag(){
  arma::frowvec wordBag(n_uv, arma::fill::zeros);
  
  // i is the current start
  wordBag(tindex[i]) = 1;
  while(uindex[i] == uindex[i+1] && vindex[i] == vindex[i+1]){
    wordBag(tindex[++i]) = 1;
  }
  
  // set i to start of next word bag or if at end, to the beginning
  if(i < n_nz){
    ++i;
  } else {
    i = 0;
  }
  return wordBag;
}

// Parse the split matrix data into armadillo matrices
struct TensorData TensorData::parse(std::string path) {
  struct TensorData td;
  std::ifstream f(path, std::ios::binary);

  if (!f.is_open()) {
    std::cout << path << std::endl;
    throw std::invalid_argument("Could not open file");
  }

  f.read(reinterpret_cast<char*>(&td.offset), sizeof(int));
  td.R = TensorData::parseTensor(f);

  return td;
}

struct Sparse3dTensor TensorData::parseTensor(std::ifstream& f) {
  int u;
  int v;
  int t;
  int nnz;
  struct Sparse3dTensor R;

  // Read metadata
  f.read(reinterpret_cast<char*>(&u), sizeof(u));
  f.read(reinterpret_cast<char*>(&v), sizeof(v));
  f.read(reinterpret_cast<char*>(&t), sizeof(t));
  f.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
  f.read(reinterpret_cast<char*>(&uv), sizeof(nnz));

  // Prepare vectors, i.e. allocate memory
  R.uindex.resize(nnz);
  R.vindex.resize(nnz);
  R.tindex.resize(nnz);

  // Read matrix entries
  f.read(reinterpret_cast<char*>(R.uindex.data()), nnz * sizeof(R.uindex[0]));
  f.read(reinterpret_cast<char*>(R.vindex.data()), nnz * sizeof(R.vindex[0]));
  f.read(reinterpret_cast<char*>(R.tindex.data()), nnz * sizeof(R.tindex[0]));
  
  // Save metadata
  R.n_nz = nnz;
  R.n_u = u;
  R.n_v = v;
  R.n_t = t;
  R.n_uv = uv;
  
  return R;
}

