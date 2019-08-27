#ifndef COORD_SOLVER_H
#define COORD_SOLVER_H

#include <RcppEigen.h>
#include "DataFunctions.h"
#include <bigmemory/MatrixAccessor.hpp>
// [[Rcpp::depends(RcppEigen, BH, bigmemory)]]

template <typename T>
class CoordSolver {

    typedef Eigen::Map<const Eigen::MatrixXd> MapMat;
    typedef Eigen::MappedSparseMatrix<double> MapSpMat;
    typedef Eigen::Map<const Eigen::VectorXd> MapVec;
    typedef Eigen::VectorXd VecXd;
    typedef Eigen::VectorXi VecXi;

protected:
    const int n;
    const int nv_total;
    MapMat y;
    double ym;
    double ys;
    T X;
    MapMat Fixed;
    MapMat XZ;
    MapVec penalty_type;
    MapVec cmult;
    const VecXd quantiles;
    MapVec ucl;
    MapVec lcl;
    const int ne;
    const int nx;
    const double tolerance;
    const int max_iterations;
    int num_passes;
    double dlx;
    VecXd penalty;
    const bool intercept;
    Eigen::Map<const Eigen::VectorXd> xm;
    Eigen::Map<Eigen::VectorXd> xv;
    Eigen::Map<const Eigen::VectorXd> xs;
    VecXd wgts_user;
    VecXd residuals;
    VecXd wgts;
    double wgts_sum;
    VecXd xbeta;
    VecXd betas;
    VecXd betas_prior;
    VecXd gradient;
    double b0;
    double b0_prior;
    const double tolerance_irls;
    Rcpp::LogicalVector strong_set;
    Rcpp::LogicalVector active_set;
    int status;
    double deviance;

public:
    // constructor (dense X matrix)
    CoordSolver(const Eigen::Ref<const Eigen::MatrixXd> & y_,
                const Eigen::Ref<const Eigen::MatrixXd> & X_,
                const Eigen::Ref<const Eigen::MatrixXd> & Fixed_,
                const Eigen::Ref<const Eigen::MatrixXd> & XZ_,
                const double * xmptr,
                double * xvptr,
                const double * xsptr,
                VecXd wgts_user_,
                bool intercept_,
                const double * penalty_type_,
                const double * cmult_,
                VecXd quantiles_,
                const double * ucl_,
                const double * lcl_,
                int ne_,
                int nx_,
                double tolerance_,
                int max_iterations_) :
    n(X_.rows()),
    nv_total(X_.cols() + Fixed_.cols() + XZ_.cols()),
    y(y_.data(), n, y_.cols()),
    ym(0.0),
    ys(1.0),
    X(X_.data(), n, X_.cols()),
    Fixed(Fixed_.data(), n, Fixed_.cols()),
    XZ(XZ_.data(), n, XZ_.cols()),
    penalty_type(penalty_type_, nv_total),
    cmult(cmult_, nv_total),
    quantiles(quantiles_),
    ucl(ucl_, nv_total),
    lcl(lcl_, nv_total),
    ne(ne_),
    nx(nx_),
    tolerance(tolerance_),
    max_iterations(max_iterations_),
    num_passes(0),
    dlx(0.0),
    penalty(2),
    intercept(intercept_),
    xm(xmptr, nv_total),
    xv(xvptr, nv_total),
    xs(xsptr, nv_total),
    wgts_user(wgts_user_),
    residuals(n),
    wgts(n),
    xbeta(n),
    betas(nv_total),
    betas_prior(nv_total),
    gradient(nv_total),
    b0(0.0),
    b0_prior(0.0),
    tolerance_irls(tolerance_),
    strong_set(nv_total, false),
    active_set(nv_total, false),
    status(0),
    deviance(0.0)
    {
        init();
    };

    // constructor (sparse X matrix)
    CoordSolver(const Eigen::Ref<const Eigen::MatrixXd> & y_,
                const MapSpMat X_,
                const Eigen::Ref<const Eigen::MatrixXd> & Fixed_,
                const Eigen::Ref<const Eigen::MatrixXd> & XZ_,
                const double * xmptr,
                double * xvptr,
                const double * xsptr,
                VecXd wgts_user_,
                bool intercept_,
                const double * penalty_type_,
                const double * cmult_,
                VecXd quantiles_,
                const double * ucl_,
                const double * lcl_,
                int ne_,
                int nx_,
                double tolerance_,
                int max_iterations_) :
        n(X_.rows()),
        nv_total(X_.cols() + Fixed_.cols() + XZ_.cols()),
        y(y_.data(), n, y_.cols()),
        ym(0.0),
        ys(1.0),
        X(X_),
        Fixed(Fixed_.data(), n, Fixed_.cols()),
        XZ(XZ_.data(), n, XZ_.cols()),
        penalty_type(penalty_type_, nv_total),
        cmult(cmult_, nv_total),
        quantiles(quantiles_),
        ucl(ucl_, nv_total),
        lcl(lcl_, nv_total),
        ne(ne_),
        nx(nx_),
        tolerance(tolerance_),
        max_iterations(max_iterations_),
        num_passes(0),
        dlx(0.0),
        penalty(2),
        intercept(intercept_),
        xm(xmptr, nv_total),
        xv(xvptr, nv_total),
        xs(xsptr, nv_total),
        wgts_user(wgts_user_),
        residuals(n),
        wgts(n),
        xbeta(n),
        betas(nv_total),
        betas_prior(nv_total),
        gradient(nv_total),
        b0(0.0),
        b0_prior(0.0),
        tolerance_irls(tolerance_),
        strong_set(nv_total, false),
        active_set(nv_total, false),
        status(0),
        deviance(0.0)
    {
        init();
    };

    // destructor
    virtual ~CoordSolver(){};

    // getters
    int getN(){return n;}
    int getNvar(){return nv_total;}
    T getX(){return X;}
    VecXd getXV(){return xv;}
    double getTolerance(){return tolerance;}
    VecXd getPenalty(){return penalty;}
    VecXd getResiduals(){return residuals;}
    VecXd getUserWeights(){return wgts_user;}
    VecXd getBetas(){return betas;}
    double getBeta0(){return b0;}
    double getDeviance(){return deviance;}
    int getNumPasses(){return num_passes;}
    VecXd getGradient(){return gradient;}
    VecXd getCmult(){return cmult;}
    Rcpp::LogicalVector getStrongSet(){return strong_set;}
    Rcpp::LogicalVector getActiveSet(){return active_set;}
    int getStatus(){return status;}
    double getYm(){return ym;}
    double getYs(){return ys;}

    // setters
    void setPenalty(double val, int pos) {penalty[pos] = val;}
    void setBetas(const Eigen::Ref<const Eigen::VectorXd> & betas_) {betas = betas_;}

    // solve GLM CD problem
    void solve() {
        while (num_passes < max_iterations) {
            coord_desc();
            update_quadratic();
            if (converged())  {
                if (check_kkt()) break;
            }
        }
        if (num_passes == max_iterations) {
            status = 1; // max iterations reached
        }
        calculate_deviance(); // ESK (Calculate deviance if converged)
    }

    // coord desc to solve weighted linear regularized regression
    void coord_desc() {
        while (num_passes < max_iterations) {
            dlx = 0.0;
            int idx = 0;
            update_beta_screen(X, penalty[0], idx);
            update_beta_screen(Fixed, penalty[0], idx);
            update_beta_screen(XZ, penalty[1], idx);
            if (intercept) update_intercept();
            ++num_passes;
            if (dlx < tolerance) break;
            while (num_passes < max_iterations) {
                dlx = 0.0;
                idx = 0;
                update_beta_active(X, penalty[0], idx);
                update_beta_active(Fixed, penalty[0], idx);
                update_beta_active(XZ, penalty[1], idx);
                if (intercept) update_intercept();
                ++num_passes;
                if (dlx < tolerance) break;
            }
        }
    }

    // coordinatewise update of features in strong set
    template <typename matType>
    void update_beta_screen(const matType & x, const double & lam, int & idx) {
        for (int k = 0; k < x.cols(); ++k, ++idx) {
            if (strong_set[idx]) {
                double gk = xs[idx] * (x.col(k).dot(residuals) - xm[idx] * residuals.sum());
                double bk = betas[idx];
                double u = gk + bk * xv[idx];
                double v = std::abs(u) - cmult[idx] * penalty_type[idx] * lam;
                if (v > 0.0) {
                    betas[idx] = std::max(lcl[idx],
                                          std::min(ucl[idx],
                                          copysign(v, u) / (xv[idx] + cmult[idx] * (1 - penalty_type[idx]) * lam)));
                }
                else {
                    betas[idx] = 0.0;
                }
                if (betas[idx] != bk) {
                    double del = betas[idx] - bk;
                    if (!active_set[idx]) {
                        active_set[idx] = true;
                    }
                    residuals -= del * xs[idx] * (x.col(k) - xm[idx]  * Eigen::VectorXd::Ones(n)).cwiseProduct(wgts);
                    dlx = std::max(dlx, xv[idx] * del * del);
                }
            }
        }
    }

    // coordinatewise update of features in active set
    template <typename matType>
    void update_beta_active(const matType & x, const double & lam, int & idx) {
        for (int k = 0; k < x.cols(); ++k, ++idx) {
            if (active_set[idx]) {
                double gk = xs[idx] * (x.col(k).dot(residuals) - xm[idx] * residuals.sum());
                double bk = betas[idx];
                double u = gk + bk * xv[idx];
                double v = std::abs(u) - cmult[idx] * penalty_type[idx] * lam;
                if (v > 0.0) {
                    betas[idx] = std::max(lcl[idx],
                                          std::min(ucl[idx],
                                          copysign(v, u) / (xv[idx] + cmult[idx] * (1 - penalty_type[idx]) * lam)));
                }
                else {
                    betas[idx] = 0.0;
                }
                if (betas[idx] != bk) {
                    double del = betas[idx] - bk;
                    residuals -= del * xs[idx] * (x.col(k) - xm[idx] * Eigen::VectorXd::Ones(n)).cwiseProduct(wgts);
                    dlx = std::max(dlx, xv[idx] * del * del);
                }
            }
        }
    }

    // update intercept
    void update_intercept(){
        double del = residuals.sum() / wgts_sum;
        b0 += del;
        residuals.array() -= del * wgts.array();
        dlx = std::max(dlx, del * del * wgts_sum);
    }

    // base initialize function
    void init(){
        xbeta = Eigen::VectorXd::Zero(n);
        betas = Eigen::VectorXd::Zero(nv_total);
        betas_prior = Eigen::VectorXd::Zero(nv_total);

        // add fixed vars to strong set
        std::fill(
            strong_set.begin() + X.cols(),
            strong_set.begin() + X.cols() + Fixed.cols(),
            true
        );
    }

    // warm start initialization given current estimates
    virtual void warm_start(const double & b0_start,
                            const Eigen::Ref<const Eigen::VectorXd> & betas_start) {

        // initialize estimates with provided values
        b0 = b0_start;
        betas = betas_start;

        // compute residuals given starting values
        int idx = 0;
        residuals.array() = wgts.array() * ((y.col(0).array() - ym) / ys - b0);
        for (int k = 0; k < X.cols(); ++k, ++idx) {
            residuals -= betas[idx] * xs[idx] * (X.col(k) - xm[idx]  * Eigen::VectorXd::Ones(n)).cwiseProduct(wgts);
        }
        for (int k = 0; k < Fixed.cols(); ++k, ++idx) {
            residuals -= betas[idx] * xs[idx] * (Fixed.col(k) - xm[idx]  * Eigen::VectorXd::Ones(n)).cwiseProduct(wgts);
        }
        for (int k = 0; k < XZ.cols(); ++k, ++idx) {
            residuals -= betas[idx] * xs[idx] * (XZ.col(k) - xm[idx]  * Eigen::VectorXd::Ones(n)).cwiseProduct(wgts);
        }

        // compute gradients given current residuals (penalized features only)
        idx = 0;
        double resids_sum = residuals.sum();
        for (int k = 0; k < X.cols(); ++k, ++idx) {
            gradient[idx] = xs[idx] * (X.col(k).dot(residuals) - xm[idx] * resids_sum);
        }
        idx += Fixed.cols();
        for (int k = 0; k < XZ.cols(); ++k, ++idx) {
            gradient[idx] = xs[idx] * (XZ.col(k).dot(residuals) - xm[idx] * resids_sum);
        }
    }

    // update quadratic approx. of likelihood function
    virtual void update_quadratic(){}

    // calculate deviance (for OLS regression deviance = sum of residuals^2)
    virtual double calculate_deviance(){
        deviance = 0;
        // First calcuate linear predictor, then extract (y - eta)^2
        xbeta.array() = 0;
        int idx = 0;
        for (int j = 0; j < X.cols(); ++j, ++idx) {
            if (strong_set[idx] && betas[idx] != 0.0) {
                xbeta += xs[idx] * (X.col(j) - xm[idx] * Eigen::VectorXd::Ones(n)) * betas[idx];
            }
        }
        for (int j = 0; j < Fixed.cols(); ++j, ++idx) {
            xbeta += xs[idx] * (Fixed.col(j) - xm[idx] * Eigen::VectorXd::Ones(n)) * betas[idx];
        }
        for (int j = 0; j < XZ.cols(); ++j, ++idx) {
            if (strong_set[idx] && betas[idx] != 0.0) {
                xbeta += xs[idx] * (XZ.col(j) - xm[idx] * Eigen::VectorXd::Ones(n)) * betas[idx];
            }
        }
        for (int i = 0; i < n; ++i) deviance += pow(y.col(0)[i] - xbeta[i], 2);
        return deviance;
    }

    // check convergence of IRLS (always converged in linear case)
    virtual bool converged() {return true;}

    // update strong set
    void update_strong(const Eigen::Ref<const VecXd> & path,
                       const Eigen::Ref<const VecXd> & path_ext,
                       const int & m,
                       const int & m2) {
        int idx = 0;
        double penalty_old = (m == 0 || (m == 1 && path[m - 1] == 9.9e35)) ? 0.0 : path[m - 1];
        double lam_diff = 2.0 * path[m] - penalty_old;
        for (int k = 0; k < X.cols(); ++k, ++idx) {
            if (!strong_set[idx]) {
                strong_set[idx] = std::abs(gradient[idx]) > lam_diff * penalty_type[idx] * cmult[idx];
            }
        }
        idx += Fixed.cols();
        if (XZ.cols() > 0) {
            if (m2 == 0) {
                std::fill(strong_set.begin() + X.cols() + Fixed.cols(), strong_set.end(), false);
                std::fill(active_set.begin() + X.cols() + Fixed.cols(), active_set.end(), false);
            }
            penalty_old = (m2 == 0 || (m2 == 1 && path[m2 - 1] == 9.9e35)) ? 0.0 : path[m2 - 1];;
            lam_diff = 2.0 * path_ext[m2] - penalty_old;
            for (int k = 0; k < XZ.cols(); ++k, ++idx) {
                if (!strong_set[idx]) {
                    strong_set[idx] = std::abs(gradient[idx]) > lam_diff * penalty_type[idx] * cmult[idx];
                }
            }
        }
    }

    // check kkt conditions
    virtual bool check_kkt() {
        int num_violations = 0;
        int idx = 0;
        double resid_sum = residuals.sum();
        for (int k = 0; k < X.cols(); ++k, ++idx) {
            if (!strong_set[idx]) {
                gradient[idx] = xs[idx] * (X.col(k).dot(residuals) - xm[idx] * resid_sum);
                if (std::abs(gradient[idx]) > penalty[0] * penalty_type[idx] * cmult[idx]) {
                    strong_set[idx] = true;
                    ++num_violations;
                }
            }
        }
        idx = idx + Fixed.cols();
        for (int k = 0; k < XZ.cols(); ++k, ++idx) {
            if (!strong_set[idx]) {
                gradient[idx] = xs[idx] * (XZ.col(k).dot(residuals) - xm[idx] * resid_sum);
                if (std::abs(gradient[idx]) > penalty[1] * penalty_type[idx] * cmult[idx]) {
                    strong_set[idx] = true;
                    ++num_violations;
                }
            }
        }
        return num_violations == 0;
    }
};

#endif // COORD_SOLVER_H
