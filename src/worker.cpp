#include <fstream>
#include <iomanip>

#include <armadillo>
#include <glog/logging.h>

#include "worker.h"

#include "matrix_data.h"

namespace mfals {

Worker::Worker(int id, std::string basepath, int k, int iterations,
               int evalrounds, int utableid, int vtableid, int ttableid)
    : id(id),
      basepath(basepath),
      k(k),
      iterations(iterations),
      evalrounds(evalrounds),
      utableid(utableid),
      vtableid(vtableid),
      ttableid(ttableid) {}

void Worker::run() {
  petuum::PSTableGroup::RegisterThread();

  std::ostringstream Upath;
  Upath << this->basepath << "/rank-" << this->id << "-train";
  struct TensorData Udata = TensorData::parse(Upath.str());
  int uoffset = Udata.offset;
  auto Ru = Udata.R;
  
  std::ostringstream Vpath;
  Vpath << this->basepath << "/rank-" << this->id << "-train";
  struct TensorData Vdata = TensorData::parse(Vpath.str());
  int voffset = Vdata.offset;
  auto Rv = Vdata.R
  
  std::ostringstream Tpath;
  Tpath << this->basepath << "/rank-" << this->id << "-train";
  struct TensorData Tdata = TensorData::parse(Tpath.str());
  int toffset = Tdata.offset;
  auto Rt = Tdata.R
  
  petuum::Table<float> Utable =
      petuum::PSTableGroup::GetTableOrDie<float>(this->utableid);
  petuum::Table<float> Vtable =
      petuum::PSTableGroup::GetTableOrDie<float>(this->vtableid);
  petuum::Table<float> Ttable =
      petuum::PSTableGroup::GetTableOrDie<float>(this->ttableid);


  // Register rows
  if (this->id == 0) {
    for (int i = 0; i < this->k; i++) {
      Utable.GetAsyncForced(i);
      Vtable.GetAsyncForced(i);
      Ttable.GetAsyncForced(i);
    }
  }

  petuum::PSTableGroup::GlobalBarrier();

  LOG(INFO) << "Randomize U, V and T";

  if (this->id == 0) {
    this->randomizetable(Utable, this->k, Ru.n_u);
    this->randomizetable(Vtable, this->k, Rv.n_v);
    this->randomizetable(Ttable, this->k, Rt.n_t);
  }

  petuum::PSTableGroup::GlobalBarrier();

  LOG(INFO) << "Fetch U, V and T on worker " << this->id;

  // Fetch U, V and T
  auto U = this->loadmat(Utable, Ru.n_u, this->k);
  auto V = this->loadmat(Vtable, Rv.n_v, this->k);
  auto T = this->loadmat(Ttable, Rt.n_t, this->k);

  LOG(INFO) << "Start optimization";

  float step = 1.0;

  for (int i = 0; i < this->iterations; i++) {
    LOG(INFO) << "Optimization round " << i << " on worker " << this->id;

    if (this->id == 0) {
      std::cout << "Round " << i + 1 << " with step length " << step
                << std::endl;
    }

    ///////
    // Compute gradient for U
    ///////
    arma::fmat Ugrad(Ru.n_u, this->k, arma::fill::zeros);
   
    // iterate over all uv pairs in Ru
    for (std::size_t i = 0; i != Ru.n_uv; ++i) {
      int uind = Ru.uindex[Ru.i];
      int vind = Ru.vindex[Ru.i];
      
      auto Td = nextWordBag();
      
      // Probably incorrect usage of armadillo ...
      arma::frowvec diff = (U.row(uind + uoffset) % V.row(vind)) * arma::inplace_trans(T)  - Td;

      for (int x = 0; x < this->k; x++) {
        Ugrad(uind, x) += 2 * V(vind, x) * arma::dot(diff, T.row(d));
      }
    }
    
    //TODO check if normalising makes sense
    Ugrad = arma::normalise(Ugrad, 2, 1);
    Ugrad = Ugrad * (-step);

    // Update U table
    for (int j = 0; j < Ugrad.n_cols; j++) {
      petuum::DenseUpdateBatch<float> batch(uoffset, Ugrad.n_rows);

      std::memcpy(batch.get_mem(), Ugrad.colptr(j),
                  Ugrad.n_rows * sizeof(float));

      Utable.DenseBatchInc(j, batch);
    }

    petuum::PSTableGroup::GlobalBarrier();

    // Fetch updated U
    U = loadmat(Utable, U.n_rows, U.n_cols);
    U = arma::clamp(U, 0.0, 5.0);
    
    
    ///////
    // Compute gradient for V
    ///////
    arma::fmat Vgrad(Rv.n_v, this->k, arma::fill::zeros);
   
    // iterate over all uv pairs in Rv
    for (std::size_t i = 0; i != Rv.n_uv; ++i) {
      int uind = Rv.uindex[Rv.i];
      int vind = Rv.vindex[Rv.i];
      
      auto Td = nextWordBag();
      
      // Probably incorrect usage of armadillo ...
      arma::frowvec diff = (U.row(uind) % V.row(vind + voffset)) * arma::inplace_trans(T)  - Td;

      for (int x = 0; x < k; x++) {
        Ugrad(uind, x) += 2 * U(uind, x) * arma::dot(diff, T.row(d));
      }
    }
    
    //TODO check if normalising makes sense
    Vgrad = arma::normalise(Vgrad, 2, 1);
    Vgrad = Vgrad * (-step);

    // Update U table
    for (int j = 0; j < Vgrad.n_cols; j++) {
      petuum::DenseUpdateBatch<float> batch(voffset, Vgrad.n_rows);

      std::memcpy(batch.get_mem(), Vgrad.colptr(j),
                  Vgrad.n_rows * sizeof(float));

      Vtable.DenseBatchInc(j, batch);
    }

    petuum::PSTableGroup::GlobalBarrier();

    // Fetch updated V
    V = loadmat(Vtable, V.n_rows, V.n_cols);
    V = arma::clamp(V, 0.0, 5.0);
    
    
    
    ///////
    // Compute gradient for T
    ///////
    arma::fmat Tgrad(Rt.n_t, k, arma::fill::zeros);
   
    // iterate over all uv pairs in Rv
    for (std::size_t i = 0; i != Rv.n_uv; ++i) {
      int uind = Rv.uindex[Rv.i];
      int vind = Rv.vindex[Rv.i];
      
      auto Td = nextWordBag();
      
      // Probably incorrect usage of armadillo ...
      arma::frowvec diff = (U.row(uind) % V.row(vind + voffset)) * arma::inplace_trans(T.rows(toffset, toffset + Rt.n_t))  - Td;

      for (int x = 0; x < k; x++) {
        Tgrad.row(x) += 2 * V(vind, x) * U(uind, x) * diff;
      }
    }
    
    //TODO check if normalising makes sense
    Vgrad = arma::normalise(Vgrad, 2, 1);
    Vgrad = Vgrad * (-step);

    // Update U table
    for (int j = 0; j < Vgrad.n_cols; j++) {
      petuum::DenseUpdateBatch<float> batch(voffset, Vgrad.n_rows);

      std::memcpy(batch.get_mem(), Vgrad.colptr(j),
                  Vgrad.n_rows * sizeof(float));

      Vtable.DenseBatchInc(j, batch);
    }

    petuum::PSTableGroup::GlobalBarrier();

    // Fetch updated V
    V = loadmat(Vtable, V.n_rows, V.n_cols);
    V = arma::clamp(V, 0.0, 5.0);
    
    
    
    

    step *= 0.9;

    // Evaluate
    if (this->evalrounds > 0 && (i + 1) % this->evalrounds == 0) {
      if (this->id == 0) {
        std::cout << "Test => ";
        this->evaltest(P, UT);
      }

      if (this->id == 0) {
        std::cout << "Training => ";
        this->eval(P, UT, Rprod, poffset, 0);
      }
    }
  }

  // Evaluate (if not evaluated in last round)
  if (this->id == 0) {
    if (this->evalrounds <= 0 || this->iterations % this->evalrounds != 0) {
      std::cout << "Test => ";
      this->evaltest(P, UT);
      std::cout << "Training => ";
      this->eval(P, UT, Rprod, poffset, 0);
    }

    P.save("out/P", arma::csv_ascii);
    UT.save("out/UT", arma::csv_ascii);
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
    std::vector<float> tmp;
    const auto& col = table.Get<petuum::DenseRow<float>>(i, &rowacc);
    col.CopyToVector(&tmp);
    std::memcpy(M.colptr(i), tmp.data(), sizeof(float) * m);
  }

  return M;
}

void Worker::evaltest(arma::fmat& P, arma::fmat& UT) {
  std::ostringstream testpath;
  testpath << this->basepath << "/test";
  std::ifstream f(testpath.str(), std::ios::binary);
  auto Rtest = TensorData::parsemat(f);

  this->eval(P, UT, Rtest, 0, 0);
}

void Worker::eval(arma::fmat& P, arma::fmat& UT, arma::sp_fmat& R,
                  int rowoffset, int coloffset) {
  float mse = 0;

  arma::sp_fmat::const_iterator start = R.begin();
  arma::sp_fmat::const_iterator end = R.end();
  for (arma::sp_fmat::const_iterator it = start; it != end; ++it) {
    int row = it.row();
    int col = it.col();

    float error =
        (*it) - arma::dot(P.row(row + rowoffset), UT.row(col + coloffset));
    mse += error * error;

    LOG(INFO) << "Product " << std::setw(7) << row + rowoffset << ", User "
              << std::setw(7) << col + coloffset << ": " << std::setw(7)
              << error << " (" << *it << ")";
  }

  std::cout << "MSE = " << mse / R.n_nonzero << std::endl;
}
}

