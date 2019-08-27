#ifndef WEIBULL_SOLVER_H
#define WEIBULL_SOLVER_H

#include "CoordSolver.h"

template <typename T>
class WeibullSolver : public CoordSolver<T> {

    typedef Eigen::VectorXd VecXd;
    typedef Eigen::VectorXi VecXi;
    typedef Eigen::Map<const Eigen::MatrixXd> MapMat;
    typedef Eigen::MappedSparseMatrix<double> MapSpMat;
    typedef Eigen::Map<const Eigen::VectorXd> MapVec;

private:
    VecXd xbeta;
    VecXd sigma;
    VecXd ftime;
    VecXd fstatus;

public:
    // constructor (dense X matrix)
    WeibullSolver(const Eigen::Ref<const Eigen::VectorXd> & y_,
                  const Eigen::Ref<const Eigen::MatrixXd> & X_,
                  const Eigen::Ref<const Eigen::MatrixXd> & Fixed_,
                  const Eigen::Ref<const Eigen::MatrixXd> & XZ_,
                  const double * xmptr,
                  double * xvptr,
                  const double * xsptr,
                  VecXd wgts_user_,
                  bool intercept_,
                  //bool aftscale_,
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
                   xbeta(this->n),
                   ftime(y_.col(0), n), //observed event times
                   fstatus(y_.col(1), n), //censoring indicator
                   sigma(this->1), // scale parameter for aft model
                   {
                       xbeta = Eigen::VectorXd::Zero(this->n);
                       sigma = 1;
                       init();
                   };

    // constructor (sparse X matrix)
    WeibullSolver(const Eigen::Ref<const Eigen::VectorXd> & y_,
                  const MapSpMat X_,
                  const Eigen::Ref<const Eigen::MatrixXd> & Fixed_,
                  const Eigen::Ref<const Eigen::MatrixXd> & XZ_,
                  const double * xmptr,
                  double * xvptr,
                  const double * xsptr,
                  VecXd wgts_user_,
                  bool intercept_,
                  //bool aftscale_,
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
                       xbeta(this->n),
                       ftime(y_.col(0), n),
                       fstatus(y_.col(1), n),
                       sigma(this->1)
                       {
                           xbeta = Eigen::VectorXd::Zero(this->n);
                           sigma = 1;
                           init();
                       };

    // destructor
    virtual ~WeibullSolver() {}

    virtual void update_aft_scale() {
        double scale.g; // define function-specific variables
        double scale.h; // define function-specific variables
        double m; // define function-specific variables
        double M; // define function-specific variables
        double mp1;

        for (int i = 0; i < this->n; ++i) {
            m = (this->y[i] - xbeta[i]) / sigma;
            M = exp(m);
            mp1 = m + 1
            scale.g += fstatus[i] * -mp1 + m * M;
            scale.h += m * (fstatus[i] - M * mp1);
        }
        scale = scale / exp(scale.g / scale.h); // ESK: Might need to check this
    }; // end update_aft_scale

    // initialize function
    virtual void init() {
        // ESK: Need to define an array for delta in the input (for survival models).
        // initialize everything with beta = 0 and sigma = 1
        // [i.e. (y - eta) / sigma = y under initial values]

        double expy; // define function-specific variables
        for (int i = 0; i < this->n; ++i) {
            expy = exp(this->y[i]);
            this->residuals[i] = this->wgts_user[i] * -(fstatus[i] - expy);
            this->wgts[i] = this->wgts_user[i] * expy;
        }

        // initial wgts (subject-specific Hessian diagonals)
        //this->wgts.array() = this->wgts_user.array() * this->y.array().exp();

        // initial residuals (subject-specific gradient values)
        //this->residuals.array() = this->wgts_user.array() * (this->delta.array() - y.array().exp);

        // add fixed vars to strong set
        std::fill(
            this->strong_set.begin() + this->X.cols(),
            this->strong_set.begin() + this->X.cols() + this->Fixed.cols(),
            true
        );

        // initial weighted sum squares x / xz cols and gradient
        int idx = 0;
        for (int k = 0; k < this->X.cols(); ++k, ++idx) {
            this->gradient[idx] = this->xs[idx] * (this->X.col(k).dot(this->residuals) - this->xm[idx] * this->residuals.sum());
            this->xv[idx] = std::pow(this->xs[idx], 2) * (this->X.col(k).cwiseProduct(this->X.col(k)) -
                2 * this->xm[idx] * this->X.col(k) + std::pow(this->xm[idx], 2) * Eigen::VectorXd::Ones(this->n)).adjoint() * this->wgts;
        }
        for (int k = 0; k < this->Fixed.cols(); ++k, ++idx) {
            this->gradient[idx] = this->xs[idx] * (this->Fixed.col(k).dot(this->residuals) - this->xm[idx] * this->residuals.sum());
            this->xv[idx] = std::pow(this->xs[idx], 2) * (this->Fixed.col(k).cwiseProduct(this->Fixed.col(k)) -
                2 * this->xm[idx] * this->Fixed.col(k) + std::pow(this->xm[idx], 2) * Eigen::VectorXd::Ones(this->n)).adjoint() * this->wgts;
        }
        for (int k = 0; k < this->XZ.cols(); ++k, ++idx) {
            this->gradient[idx] = this->xs[idx] * (this->XZ.col(k).dot(this->residuals) - this->xm[idx] * this->residuals.sum());
            this->xv[idx] = std::pow(this->xs[idx], 2) * (this->XZ.col(k).cwiseProduct(this->XZ.col(k)) -
                2 * this->xm[idx] * this->XZ.col(k) + std::pow(this->xm[idx], 2) * Eigen::VectorXd::Ones(this->n)).adjoint() * this->wgts;
        }
    } // end init()

    // warm start initialization given current estimates
    virtual void warm_start(const Eigen::Ref<const Eigen::VectorXd> & y,
                            const double & b0_start,
                            const Eigen::Ref<const Eigen::VectorXd> & betas_start) {

        // initialize estimates with provided values
        this->b0 = b0_start;
        this->betas = betas_start;

        // update residuals, working response, weighted sum squares X / XZ
        update_quadratic();

        // update partial gradients given current residuals
        int idx = 0;
        for (int k = 0; k < this->X.cols(); ++k, ++idx) {
            this->gradient[idx] = this->xs[idx] * (this->X.col(k).dot(this->residuals) - this->xm[idx] * this->residuals.sum());
        }
        idx += this->Fixed.cols();
        for (int k = 0; k < this->XZ.cols(); ++k, ++idx) {
            this->gradient[idx] = this->xs[idx] * (this->XZ.col(k).dot(this->residuals) - this->xm[idx] * this->residuals.sum());
        }
    } // end warm_start()

    // update quadratic approx. of log-likelihood (subject-specific gradients & hessians)
    virtual void update_quadratic() {

        double m, M; // initialize function-dependent variables

        // compute linear predictor (X * beta)
        xbeta.array() = this->b0;
        int idx = 0;
        for (int j = 0; j < this->X.cols(); ++j, ++idx) {
            if (this->strong_set[idx] && this->betas[idx] != 0.0) {
                xbeta += this->xs[idx] * (this->X.col(j) - this->xm[idx] * Eigen::VectorXd::Ones(this->n)) * this->betas[idx];
            }
        }
        for (int j = 0; j < this->Fixed.cols(); ++j, ++idx) {
            xbeta += this->xs[idx] * (this->Fixed.col(j) - this->xm[idx] * Eigen::VectorXd::Ones(this->n)) * this->betas[idx];
        }
        for (int j = 0; j < this->XZ.cols(); ++j, ++idx) {
            if (this->strong_set[idx] && this->betas[idx] != 0.0) {
                xbeta += this->xs[idx] * (this->XZ.col(j) - this->xm[idx] * Eigen::VectorXd::Ones(this->n)) * this->betas[idx];
            }
        }

        // compute subject-specific quantities (wgts & residuals)
        for (int i = 0; i < this->n; ++i) {
            m = (this->y[i] - xbeta[i]) / sigma;
            M = exp(m);
            this->residuals[i] = this->wgts_user[i] * -(fstatus[i] - M) / sigma;
            this->wgts[i] = this->wgts_user[i] * M / pow(sigma, 2);
        }

        // update weights (subject-specific Hessian diagonals)
        //this->wgts.array() = this->wgts_user.array() * wgts.array();
        this->wgts_sum = this->wgts.sum();

        // update residuals (subject-specific gradient)
        //this->residuals.array() = this->wgts_user.array() * (this->delta.array() - prob.array());

        // update weighted sum squares x / xz cols
        idx = 0;
        for (int k = 0; k < this->X.cols(); ++k, ++idx) {
            if (this->strong_set[idx])
                this->xv[idx] = std::pow(this->xs[idx], 2) * (this->X.col(k).cwiseProduct(this->X.col(k)) -
                    2 * this->xm[idx] * this->X.col(k) + std::pow(this->xm[idx], 2) * Eigen::VectorXd::Ones(this->n)).adjoint() * this->wgts;
        }
        for (int k = 0; k < this->Fixed.cols(); ++k, ++idx) {
            if (this->strong_set[idx])
                this->xv[idx] = std::pow(this->xs[idx], 2) * (this->Fixed.col(k).cwiseProduct(this->Fixed.col(k)) -
                    2 * this->xm[idx] * this->Fixed.col(k) + std::pow(this->xm[idx], 2) * Eigen::VectorXd::Ones(this->n)).adjoint() * this->wgts;
        }
        for (int k = 0; k < this->XZ.cols(); ++k, ++idx) {
            if (this->strong_set[idx])
                this->xv[idx] = std::pow(this->xs[idx], 2) * (this->XZ.col(k).cwiseProduct(this->XZ.col(k)) -
                    2 * this->xm[idx] * this->XZ.col(k) + std::pow(this->xm[idx], 2) * Eigen::VectorXd::Ones(this->n)).adjoint() * this->wgts;
        }
    } // End: update_quadratic()

    // check convergence of IRLS
    // ESK: This doesn't change
    virtual bool converged() {
        bool converged_outer = true;
        if (this->wgts.sum() >= 1e-9) {
            if (this->wgts.sum() * std::pow(this->b0 - this->b0_prior, 2) > this->tolerance_irls) {
                converged_outer = false;
            }
            else {
                for (int k = 0; k < this->nv_total; ++k) {
                    if (this->strong_set[k] && this->xv[k] * std::pow(this->betas[k] - this->betas_prior[k], 2) > this->tolerance_irls) {
                        converged_outer = false;
                        break;
                    }
                }
            }
        }
        this->betas_prior = this->betas;
        this->b0_prior = this->b0;
        return converged_outer;
    } // end converged()

    // check kkt conditions
    // ESK: This doesn't change
    virtual bool check_kkt() {
        int num_violations = 0;
        int idx = 0;
        double resid_sum = this->residuals.sum();
        for (int k = 0; k < this->X.cols(); ++k, ++idx) {
            if (!this->strong_set[idx]) {
                this->gradient[idx] = this->xs[idx] * (this->X.col(k).dot(this->residuals) - this->xm[idx] * resid_sum);
                if (std::abs(this->gradient[idx]) > this->penalty[0] * this->penalty_type[idx] * this->cmult[idx]) {
                    this->strong_set[idx] = true;
                    this->xv[idx] = std::pow(this->xs[idx], 2) * (this->X.col(k).cwiseProduct(this->X.col(k)) -
                        2 * this->xm[idx] * this->X.col(k) + std::pow(this->xm[idx], 2) * Eigen::VectorXd::Ones(this->n)).adjoint() * this->wgts;
                    ++num_violations;
                }
            }
        }
        idx += this->Fixed.cols();
        for (int k = 0; k < this->XZ.cols(); ++k, ++idx) {
            if (!this->strong_set[idx]) {
                this->gradient[idx] = this->xs[idx] * (this->XZ.col(k).dot(this->residuals) - this->xm[idx] * resid_sum);
                if (std::abs(this->gradient[idx]) > this->penalty[0] * this->penalty_type[idx] * this->cmult[idx]) {
                    this->strong_set[idx] = true;
                    this->xv[idx] = std::pow(this->xs[idx], 2) * (this->XZ.col(k).cwiseProduct(this->XZ.col(k)) -
                        2 * this->xm[idx] * this->XZ.col(k) + std::pow(this->xm[idx], 2) * Eigen::VectorXd::Ones(this->n)).adjoint() * this->wgts;
                    ++num_violations;
                }
            }
        }
        return num_violations == 0;
    } // end check_kkt()

    virtual void get_deviance() {
        double m;
        double M;
        double dev = 0;

        for (int i = 0; i < this->n; ++i) {
            m = (this->y[i] - xbeta[i]) / sigma;
            M = exp(m);
            dev += (fstatus[i] * (-log(sigma) + m) - M); // calculating log-likelihood
        }
        dev = -2 * dev; // this is deviance
    } // end get_deviance

};

#endif // WEIBULL_SOLVER_H
