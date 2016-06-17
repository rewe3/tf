#include <fstream>
#include <iomanip>

#include <armadillo>
#include <glog/logging.h>

#include "worker.h"

#include "tensor_data.h"

namespace tfals {

Worker::Worker(int id, std::string basepath, int rank, int iterations,
               int evalrounds, int usertableid, int prodtableid, int wordstableid)
    : id(id),
      basepath(basepath),
      rank(rank),
      iterations(iterations),
      evalrounds(evalrounds),
      usertableid(usertableid),
      prodtableid(prodtableid),
      wordtableid(wordtableid) {}

void Worker::run() {
  petuum::PSTableGroup::RegisterThread();

  std::ostringstream userpath;
  userpath << this->basepath << "/_user_train" << this->id;
  struct TensorData userdata = TensorData::parse(userpath.str());
  int useroffset = userdata.offset;
  auto Ruser = userdata.R;
  
  std::ostringstream prodpath;
  prodpath << this->basepath << "/_prod_train" << this->id;
  struct TensorData proddata = TensorData::parse(prodpath.str());
  int prodoffset = proddata.offset;
  auto Rprod = proddata.R;
  
  std::ostringstream wordpath;
  wordpath << this->basepath << "/_word_train" << this->id;
  struct TensorData worddata = TensorData::parse(wordpath.str());
  int wordoffset = worddata.offset;
  auto Rwords = worddata.R;
  
  petuum::Table<float> usertable =
      petuum::PSTableGroup::GetTableOrDie<float>(this->usertableid);
  petuum::Table<float> prodtable =
      petuum::PSTableGroup::GetTableOrDie<float>(this->prodtableid);
  petuum::Table<float> wordtable =
      petuum::PSTableGroup::GetTableOrDie<float>(this->wordtableid);

  std::cout << id << " got tables" << std::endl;
  // Register rows
  if (this->id == 0) {
    for (int i = 0; i < this->rank; i++) {
      usertable.GetAsyncForced(i);
      prodtable.GetAsyncForced(i);
      wordtable.GetAsyncForced(i);
    }
  }

  petuum::PSTableGroup::GlobalBarrier();
  
  std::cout << id << " past 1st barrier" << std::endl;
  
  LOG(INFO) << "Randomize U, P and T";

  if (id == 0) {
    std::cout << Rwords.n_rows << " " << Rwords.n_cols << " " << Ruser.n_words << std::endl;
    randomizetable(usertable, rank, Rwords.n_rows);
    randomizetable(prodtable, rank, Rwords.n_cols);
    randomizetable(wordtable, rank, Ruser.n_words);
  }
  std::cout << id << " before 2nd barrier" << std::endl;
  //petuum::PSTableGroup::GlobalBarrier();
  std::cout << id << " past 2nd barrier" << std::endl;
  
  LOG(INFO) << "Fetch U, P and T on worker " << id;
  // Fetch U, P and T
  std::cout << id <<" U size " << Rwords.n_rows << " " << rank << std::endl;
  auto U = this->loadmat(usertable, Rwords.n_rows, rank);
  std::cout << id <<" P size " << Rwords.n_cols << " " << rank << std::endl;
  auto P = this->loadmat(prodtable, Rwords.n_cols, rank);
  std::cout << id <<" T size " << Ruser.n_words << " " << rank << std::endl;
  auto T = this->loadmat(wordtable, Ruser.n_words, rank);
  arma::inplace_trans(T);

  LOG(INFO) << "Start optimization";

  std::cout << id << " start optimizing" << std::endl;
  float step = 1.0;

  for (int i = 0; i < this->iterations; i++) {
    LOG(INFO) << "Optimization round " << i << " on worker " << id;

    if (id == 0) {
      std::cout << "Round " << i + 1 << " with step length " << step
                << std::endl;
    }

    ///////
    // Compute gradient for U
    ///////
    arma::fmat Ugrad(Ruser.n_rows, rank, arma::fill::zeros);
   
    // iterate over all up pairs in Ruser
    for (std::size_t i = 0; i != Ruser.n_nz; ++i) {
      int userind = Ruser.rows[i];
      int prodind = Ruser.cols[i];
      
      auto wordbag = Ruser.getWordBagAt(i);
      
      // Probably incorrect usage of armadillo ...
      arma::frowvec diff = (U.row(userind + useroffset) % P.row(prodind)) * T - wordbag;

      Ugrad.row(userind) += 2 * P.row(prodind) % (diff * T.t());
    }
    
    //TODO check if normalising makes sense
    Ugrad = arma::normalise(Ugrad, 2, 1);
    Ugrad = Ugrad * (-step);

    // Update U table
    for (int j = 0; j < Ugrad.n_cols; j++) {
      petuum::DenseUpdateBatch<float> batch(useroffset, Ugrad.n_rows);

      std::memcpy(batch.get_mem(), Ugrad.colptr(j),
                  Ugrad.n_rows * sizeof(float));

      usertable.DenseBatchInc(j, batch);
    }

    petuum::PSTableGroup::GlobalBarrier();

    // Fetch updated U
    U = this->loadmat(usertable, U.n_rows, U.n_cols);
    U = arma::clamp(U, 0.0, 5.0);
    
    
    ///////
    // Compute gradient for P
    ///////
    arma::fmat Pgrad(Rprod.n_rows, rank, arma::fill::zeros);
   
    // iterate over all up pairs in Rprod
    for (std::size_t i = 0; i != Rprod.n_nz; ++i) {
      int userind = Rprod.rows[i];
      int prodind = Rprod.cols[i];
      
      auto wordbag = Rprod.getWordBagAt(i);
      
      // Probably incorrect usage of armadillo ...
      arma::frowvec diff = (U.row(userind) % P.row(prodind + prodoffset)) * T - wordbag;

      Pgrad.row(prodind) += 2 * U.row(userind) % (diff * T.t());
    }
    
    //TODO check if normalising makes sense
    Pgrad = arma::normalise(Pgrad, 2, 1);
    Pgrad = Pgrad * (-step);

    // Update P table
    for (int j = 0; j < Pgrad.n_cols; j++) {
      petuum::DenseUpdateBatch<float> batch(prodoffset, Pgrad.n_rows);

      std::memcpy(batch.get_mem(), Pgrad.colptr(j),
                  Pgrad.n_rows * sizeof(float));

      prodtable.DenseBatchInc(j, batch);
    }

    petuum::PSTableGroup::GlobalBarrier();
  
    // Fetch updated P
    P = this->loadmat(prodtable, P.n_rows, P.n_cols);
    P = arma::clamp(P, 0.0, 5.0);
    
    
    
    ///////
    // Compute gradient for T
    ///////
    arma::fmat Tgrad(rank, Rwords.n_words, arma::fill::zeros);
   
    // iterate over all uv pairs in Rwords
    for (std::size_t i = 0; i != Rwords.n_nz; ++i) {
      int userind = Rprod.rows[i];
      int prodind = Rprod.cols[i];
      
      auto wordbag = Rwords.getWordBagAt(i);
      
      // Probably incorrect usage of armadillo ...
      arma::frowvec diff = (U.row(userind) % P.row(prodind)) * T.cols(wordoffset, wordoffset + Rwords.n_words) - wordbag;

      for (int x = 0; x < rank; x++) {
        Tgrad.row(x) += 2 * P(prodind, x) * U(userind, x) * diff;
      }
    }
    
    //TODO check if normalising makes sense
    Tgrad = arma::normalise(Tgrad, 2, 1);
    Tgrad = Tgrad * (-step);

    // Update T table
    for (int j = 0; j < Tgrad.n_cols; j++) {
      petuum::DenseUpdateBatch<float> batch(prodoffset, Tgrad.n_rows);

      std::memcpy(batch.get_mem(), Tgrad.colptr(j),
                  Tgrad.n_rows * sizeof(float));

      prodtable.DenseBatchInc(j, batch);
    }

    petuum::PSTableGroup::GlobalBarrier();

    // Fetch updated T
    T = this->loadmat(wordtable, T.n_rows, T.n_cols);
    T = arma::clamp(T, 0.0, 5.0);
    arma::inplace_trans(T);

    step *= 0.9;

    // Evaluate
    if (this->evalrounds > 0 && (i + 1) % this->evalrounds == 0) {
      if (id == 0) {
        std::cout << "Test => ";
        this->evaltest(U, P, T);
      }

      if (id == 0) {
        std::cout << "Training => ";
        eval(U, P, T, Ruser, useroffset, 0);
      }
    }
  }

  // Evaluate (if not evaluated in last round)
  if (id == 0) {
    if (this->evalrounds <= 0 || this->iterations % this->evalrounds != 0) {
      std::cout << "Test => ";
      this->evaltest(U, P, T);
      std::cout << "Training => ";
      eval(U, P, T, Ruser, useroffset, 0);
    }

    U.save("out/U", arma::csv_ascii);
    P.save("out/P", arma::csv_ascii);
    P.save("out/T", arma::csv_ascii);
  }

  LOG(INFO) << "Shutdown worker " << this->id;

  petuum::PSTableGroup::DeregisterThread();
}

// Initialize table as an m*n matrix with random entries
void Worker::randomizetable(petuum::Table<float>& table, int m, int n) {
  arma::fvec vec(n);

  for (int i = 0; i < m; i++) {
    vec.randn();
    vec = arma::abs(vec);

    petuum::DenseUpdateBatch<float> batch(0, n);
    std::memcpy(batch.get_mem(), vec.memptr(), n * sizeof(float));

    table.DenseBatchInc(i, batch);
  }
}

arma::fmat Worker::loadmat(petuum::Table<float>& table, int m, int n) {
  arma::fmat M(m, n);
  petuum::RowAccessor rowacc;

  for (int i = 0; i < n; i++) {
    try{
    std::vector<float> tmp;
    std::cout << "It " << i << std::endl;
    const auto& col = table.Get<petuum::DenseRow<float>>(i, &rowacc);
    col.CopyToVector(&tmp);
    std::cout << tmp.size() << std::endl;
    std::memcpy(M.colptr(i), tmp.data(), sizeof(float) * m);
    } catch(exception& e){
    }
  }
  return M;
}

void Worker::evaltest(arma::fmat& U, arma::fmat& P, arma::fmat& T) {
  /*std::ostringstream testpath;
  testpath << basepath << "/test";
  std::ifstream f(testpath.str(), std::ios::binary);
  auto Rtest = TensorData::parseTensor(f);

  eval(U, P, T, Rtest, 0, 0);*/
}

void Worker::eval(arma::fmat& U, arma::fmat& P, arma::fmat& T, Sparse3dTensor& R,
                  int useroffset, int prodoffset) {
  float mse = 0;

  for (size_t i = 0; i != R.n_nz; ++i) {
    int userind = R.rows[i];
    int prodind = R.cols[i];

    arma::frowvec error =
        R.getWordBagAt(i) - (U.row(userind + useroffset) % P.row(prodind + prodoffset)) * T;
    mse += arma::norm(error);

    LOG(INFO) << "User " << std::setw(7) << userind + useroffset << ", Product "
              << std::setw(7) << prodind + prodoffset << ": " << std::setw(7)
              << error; //<< " (" << R.getWordBagAt(i) << ")";
  }

  std::cout << "MSE = " << mse / R.n_nz << std::endl;
}
}

