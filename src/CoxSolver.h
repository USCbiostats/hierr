#ifndef COX_SOLVER_H
#define COX_SOLVER_H

#include <vector>
#include "CoordSolver.h"

using namespace Eigen;
using namespace std;

template <typename T>
class CoxSolver : public CoordSolver<T> {

    typedef Eigen::VectorXd VecXd;
    typedef Eigen::VectorXi VecXi;
    typedef Eigen::Map<const Eigen::MatrixXd> MapMat;
    typedef Eigen::MappedSparseMatrix<double> MapSpMat;
    typedef Eigen::Map<const Eigen::VectorXd> MapVec;

private:
    VecXd eta;
    vector<double> D;
    vector<double> d;
    VecXi ri;
    VecXi ck;
    int m;
    using CoordSolver<T>::n;
    using CoordSolver<T>::nv_total;
    using CoordSolver<T>::intercept;
    using CoordSolver<T>::wgts;
    using CoordSolver<T>::wgts_user;
    using CoordSolver<T>::wgts_sum;
    using CoordSolver<T>::y;
    using CoordSolver<T>::X;
    using CoordSolver<T>::Fixed;
    using CoordSolver<T>::XZ;
    using CoordSolver<T>::xm;
    using CoordSolver<T>::xs;
    using CoordSolver<T>::xv;
    using CoordSolver<T>::residuals;
    using CoordSolver<T>::gradient;
    using CoordSolver<T>::betas;
    using CoordSolver<T>::b0;
    using CoordSolver<T>::betas_prior;
    using CoordSolver<T>::b0_prior;
    using CoordSolver<T>::tolerance_irls;
    using CoordSolver<T>::penalty;
    using CoordSolver<T>::penalty_type;
    using CoordSolver<T>::cmult;
    using CoordSolver<T>::strong_set;


public:
    // constructor (dense X matrix)
    CoxSolver(const Eigen::Ref<const Eigen::MatrixXd> & y_,
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
              CoordSolver<T>(y_,
                             X_,
                             Fixed_,
                             XZ_,
                             xmptr,
                             xvptr,
                             xsptr,
                             wgts_user_,
                             intercept_,
                             penalty_type_,
                             cmult_,
                             quantiles_,
                             ucl_,
                             lcl_,
                             ne_,
                             nx_,
                             tolerance_,
                             max_iterations_),
                             eta(n)
                             {
                                 init();
                             };

    // constructor (sparse X matrix)
    CoxSolver(const Eigen::Ref<const Eigen::MatrixXd> & y_,
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
              CoordSolver<T>(y_,
                             X_,
                             Fixed_,
                             XZ_,
                             xmptr,
                             xvptr,
                             xsptr,
                             wgts_user_,
                             intercept_,
                             penalty_type_,
                             cmult_,
                             quantiles_,
                             ucl_,
                             lcl_,
                             ne_,
                             nx_,
                             tolerance_,
                             max_iterations_),
                             eta(n)
                             {
                                 init();
                             };

    // destructor
    virtual ~CoxSolver() {}

    // initialize function
    void init() {

        // get unique event time, D, and ties at event time, d
        int events = y(0, 1);
        for (int k = 1; k < n; ++k) {
            if (y(k, 0) != y(k - 1, 0)) {
                if (events) {
                    D.push_back(y(k - 1, 0));
                    d.push_back(events);
                }
                events = 0;
            }
            events += y(k, 1);
        }
        if (events) {
            D.push_back(y(n - 1, 0));
            d.push_back(events);
        }
        m = D.size();

        // get ck, and ri, risk sets
        int ck_prime = 0;
        ri.setZero(m + 1);
        ck.setZero(n + 1);
        for (int k = 0; k < n; ++k) {
            ck[k + 1] = ck_prime;
            for (int j = ck_prime; j < m; ++j) {
                if (D[j] > y(k, 0)) break;
                ck[k + 1] += 1;
                ri[ck[k + 1] - 1] = n - k;
                ck_prime = ck[k + 1];
            }
        }

        // initial wgts and residuals
        eta.setZero(n);
        VecXd exp_eta = VectorXd::Ones(n);
        double sum_exp_eta_prime = 0;
        VectorXd sum_exp_eta(m);
        for (int i = m - 1; i >= 0; --i) {
            sum_exp_eta[i] = sum_exp_eta_prime + exp_eta.segment(n - ri[i], ri[i] - ri[i + 1]).sum();
            sum_exp_eta_prime = sum_exp_eta[i];
        }

        double u_prime = 0;
        double u2_prime = 0;
        for (int k = 0; k < n; ++k) {
            if (ck[k + 1] != ck[k]) {
                u_prime += d[ck[k + 1] - 1] / sum_exp_eta[ck[k + 1] - 1];
                u2_prime += d[ck[k + 1] - 1] / (sum_exp_eta[ck[k + 1] - 1] * sum_exp_eta[ck[k + 1] - 1]);
            }
            wgts[k] = wgts_user[k] * exp_eta[k] * (u_prime - exp_eta[k] * u2_prime);
            residuals[k] = wgts_user[k] * (y(k, 1) - exp_eta[k] * u_prime);
        }
        wgts_sum = wgts.sum();

        // initial weighted sum squares x / xz cols and gradient
        int idx = 0;
        for (int k = 0; k < X.cols(); ++k, ++idx) {
            gradient[idx] = xs[idx] * (X.col(k).dot(residuals) - xm[idx] * residuals.sum());
            xv[idx] = std::pow(xs[idx], 2) * (X.col(k).cwiseProduct(X.col(k)) - 2 * xm[idx] * X.col(k) + std::pow(xm[idx], 2) * Eigen::VectorXd::Ones(n)).adjoint() * wgts;
        }
        for (int k = 0; k < Fixed.cols(); ++k, ++idx) {
            gradient[idx] = xs[idx] * (Fixed.col(k).dot(residuals) - xm[idx] * residuals.sum());
            xv[idx] = std::pow(xs[idx], 2) * (Fixed.col(k).cwiseProduct(Fixed.col(k)) - 2 * xm[idx] * Fixed.col(k) + std::pow(xm[idx], 2) * Eigen::VectorXd::Ones(n)).adjoint() * wgts;
        }
        for (int k = 0; k < XZ.cols(); ++k, ++idx) {
            gradient[idx] = xs[idx] * (XZ.col(k).dot(residuals) - xm[idx] * residuals.sum());
            xv[idx] = std::pow(xs[idx], 2) * (XZ.col(k).cwiseProduct(XZ.col(k)) - 2 * xm[idx] * XZ.col(k) + std::pow(xm[idx], 2) * Eigen::VectorXd::Ones(n)).adjoint() * wgts;
        }
    }

    // warm start initialization given current estimates
    virtual void warm_start(const double & b0_start,
                            const Eigen::Ref<const Eigen::VectorXd> & betas_start) {

        // initialize estimates with provided values
        b0 = b0_start;
        betas = betas_start;

        // update residuals, working response, weighted sum squres X / XZ
        update_quadratic();

        // update gradients given current residuals
        int idx = 0;
        for (int k = 0; k < X.cols(); ++k, ++idx) {
            gradient[idx] = xs[idx] * (X.col(k).dot(residuals) - xm[idx] * residuals.sum());
        }
        idx += Fixed.cols();
        for (int k = 0; k < XZ.cols(); ++k, ++idx) {
            gradient[idx] = xs[idx] * (XZ.col(k).dot(residuals) - xm[idx] * residuals.sum());
        }
    }

    // update quadratic approx. of log-likelihood
    virtual void update_quadratic() {
        // compute linear predictor (X * beta)
        eta.array() = 0.0;
        int idx = 0;
        for (int j = 0; j < X.cols(); ++j, ++idx) {
            if (strong_set[idx] && betas[idx] != 0.0) {
                eta += xs[idx] * (X.col(j) - xm[idx] * Eigen::VectorXd::Ones(n)) * betas[idx];
            }
        }
        for (int j = 0; j < Fixed.cols(); ++j, ++idx) {
            eta += xs[idx] * (Fixed.col(j) - xm[idx] * Eigen::VectorXd::Ones(n)) * betas[idx];
        }
        for (int j = 0; j < XZ.cols(); ++j, ++idx) {
            if (strong_set[idx] && betas[idx] != 0.0) {
                eta += xs[idx] * (XZ.col(j) - xm[idx] * Eigen::VectorXd::Ones(n)) * betas[idx];
            }
        }
        VecXd exp_eta = eta.array().exp();
        double sum_exp_eta_prime = 0;
        VectorXd sum_exp_eta(m);
        for (int i = m - 1; i >= 0; --i) {
            sum_exp_eta[i] = sum_exp_eta_prime + exp_eta.segment(n - ri[i], ri[i] - ri[i + 1]).sum();
            sum_exp_eta_prime = sum_exp_eta[i];
        }

        double u_prime = 0;
        double u2_prime = 0;
        for (int k = 0; k < n; ++k) {
            if (ck[k + 1] != ck[k]) {
                u_prime += d[ck[k + 1] - 1] / sum_exp_eta[ck[k + 1] - 1];
                u2_prime += d[ck[k + 1] - 1] / (sum_exp_eta[ck[k + 1] - 1] * sum_exp_eta[ck[k + 1] - 1]);
            }
            wgts[k] = wgts_user[k] * exp_eta[k] * (u_prime - exp_eta[k] * u2_prime);
            residuals[k] = wgts_user[k] * (y(k, 1) - exp_eta[k] * u_prime);
        }
        wgts_sum = wgts.sum();

        // update weighted sum squares x / xz cols
        idx = 0;
        for (int k = 0; k < X.cols(); ++k, ++idx) {
            if (strong_set[idx])
                xv[idx] = std::pow(xs[idx], 2) * (X.col(k).cwiseProduct(X.col(k)) - 2 * xm[idx] * X.col(k) + std::pow(xm[idx], 2) * Eigen::VectorXd::Ones(n)).adjoint() * wgts;
        }
        for (int k = 0; k < Fixed.cols(); ++k, ++idx) {
            if (strong_set[idx])
                xv[idx] = std::pow(xs[idx], 2) * (Fixed.col(k).cwiseProduct(Fixed.col(k)) - 2 * xm[idx] * Fixed.col(k) + std::pow(xm[idx], 2) * Eigen::VectorXd::Ones(n)).adjoint() * wgts;
        }
        for (int k = 0; k < XZ.cols(); ++k, ++idx) {
            if (strong_set[idx])
                xv[idx] = std::pow(xs[idx], 2) * (XZ.col(k).cwiseProduct(XZ.col(k)) - 2 * xm[idx] * XZ.col(k) + std::pow(xm[idx], 2) * Eigen::VectorXd::Ones(n)).adjoint() * wgts;
        }
    }

    // check convergence of IRLS
    virtual bool converged() {
        bool converged_outer = true;
        for (int k = 0; k < nv_total; ++k) {
            if (strong_set[k] && xv[k] * std::pow(betas[k] - betas_prior[k], 2) > tolerance_irls) {
                converged_outer = false;
                break;
            }
        }
        betas_prior = betas;
        return converged_outer;
    }
};

#endif // COX_SOLVER_H
