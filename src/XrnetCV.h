#ifndef XRNET_CV_H
#define XRNET_CV_H

#include <RcppEigen.h>
#include <unordered_map>
#include "Xrnet.h"

template <typename TX, typename TZ>
class XrnetCV : public Xrnet<TX, TZ>  {

    typedef Eigen::VectorXd VecXd;
    typedef Eigen::MatrixXd MatXd;
    typedef Eigen::Map<const Eigen::VectorXd> MapVec;
    typedef Eigen::Map<const Eigen::MatrixXd> MapMat;
    typedef Eigen::MappedSparseMatrix<double> MapSpMat;
    typedef double (*lossPtr)(const Eigen::Ref<const Eigen::VectorXd> &,
                              const Eigen::Ref<const Eigen::VectorXd> &,
                              const Eigen::Ref<const Eigen::VectorXi> &);

protected:
    Eigen::Map<const Eigen::VectorXi> test_idx;
    TX X;
    MapVec y;
    VecXd error_mat;
    lossPtr loss_func;

public:
    // constructor (dense X)
    XrnetCV(const int & n_,
            const int & nv_x_,
            const int & nv_fixed_,
            const int & nv_ext_,
            const int & nv_total_,
            const bool & intr_,
            const bool & intr_ext_,
            const TZ ext_,
            const double * xmptr,
            const double * centptr,
            const double * xsptr,
            const double & ym_,
            const double & ys_,
            const int & num_penalty_,
            const std::string & family_,
            const std::string & user_loss_,
            const Eigen::Ref<const Eigen::VectorXi> & test_idx_,
            const Eigen::Ref<const Eigen::MatrixXd> & X_,
            const Eigen::Ref<const Eigen::VectorXd> & y_) :
    Xrnet<TX, TZ>(
            n_,
            nv_x_,
            nv_fixed_,
            nv_ext_,
            nv_total_,
            intr_,
            intr_ext_,
            ext_,
            xmptr,
            centptr,
            xsptr,
            ym_,
            ys_,
            1),
            test_idx(test_idx_.data(), test_idx_.size()),
            X(X_.data(), n_, X_.cols()),
            y(y_.data(), n_)
            {
                error_mat = Eigen::VectorXd::Zero(num_penalty_);
                loss_func = select_loss(family_, user_loss_);
            };

    // constructor (sparse X)
    XrnetCV(const int & n_,
            const int & nv_x_,
            const int & nv_fixed_,
            const int & nv_ext_,
            const int & nv_total_,
            const bool & intr_,
            const bool & intr_ext_,
            const TZ ext_,
            const double * xmptr,
            const double * centptr,
            const double * xsptr,
            const double & ym_,
            const double & ys_,
            const int & num_penalty_,
            const std::string & family_,
            const std::string & user_loss_,
            const Eigen::Ref<const Eigen::VectorXi> & test_idx_,
            const MapSpMat X_,
            const Eigen::Ref<const Eigen::VectorXd> & y_) :
        Xrnet<TX, TZ>(
                n_,
                nv_x_,
                nv_fixed_,
                nv_ext_,
                nv_total_,
                intr_,
                intr_ext_,
                ext_,
                xmptr,
                centptr,
                xsptr,
                ym_,
                ys_,
                1),
                test_idx(test_idx_.data(), test_idx_.size()),
                X(X_),
                y(y_.data(), n_)
                {
                    error_mat = Eigen::VectorXd::Zero(num_penalty_);
                    loss_func = select_loss(family_, user_loss_);
                };

    // destructor
    virtual ~XrnetCV(){};

    // getters
    MatXd get_error_mat(){return error_mat;};

    // save results for single penalty
    virtual void add_results(double b0, VecXd coef, const int & idx) {

        // unstandardize variables
        coef = this->ys * coef.cwiseProduct(this->xs);
        b0 *= this->ys;

        // get external coefficients
        if (this->nv_ext > 0) {
            this->alphas.col(0) = coef.tail(this->nv_ext);
        }

        // unstandardize predictors w/ external data (x)
        if (this->nv_ext + this->intr_ext > 0) {
            VecXd z_alpha = Eigen::VectorXd::Zero(this->nv_x);
            if (this->intr_ext) {
                z_alpha.array() += coef[this->nv_x + this->nv_fixed];
            }
            if (this->nv_ext > 0) {
                z_alpha += this->ext * coef.tail(this->nv_ext);
            }
            this->betas.col(0) = z_alpha.cwiseProduct(this->xs.head(this->nv_x)) + coef.head(this->nv_x);
        }
        else {
            this->betas.col(0) = coef.head(this->nv_x);
        }

        // unstandardize predictors w/o external data (fixed)
        if (this->nv_fixed > 0) {
            this->gammas.col(0) = coef.segment(this->nv_x, this->nv_fixed);
        }

        // compute 1st level intercept
        if (this->intr) {
            this->beta0[0] = (this->ym + b0) - this->cent.head(this->nv_x).dot(this->betas.col(0));
            if (this->nv_fixed > 0) {
                this->beta0[0] -= this->cent.segment(this->nv_x, this->nv_fixed).dot(this->gammas.col(0));
            }
        }

        // compute predicted values
        VecXd yhat = Eigen::VectorXd::Constant(this->n, this->beta0[0]);
        yhat += X * this->betas.sparseView();

        // compute error for test data
        error_mat[idx] = loss_func(y, yhat, test_idx);
    }

    // returns pointer to appropriate loss function based on family / user_loss
    lossPtr select_loss(const std::string & family, const std::string & user_loss) {

        std::unordered_map<std::string, lossPtr> lossMap;
        lossMap.insert(std::make_pair<std::string, lossPtr>("mse", mean_squared_error));
        lossMap.insert(std::make_pair<std::string, lossPtr>("mae", mean_absolute_error));
        lossMap.insert(std::make_pair<std::string, lossPtr>("auc", auc));
        lossMap.insert(std::make_pair<std::string, lossPtr>("deviance_binomial", deviance_binomial));

        lossPtr loss_func = nullptr;
        if (user_loss == "default")
        {
            if (family == "gaussian") {
                loss_func = mean_squared_error;
            }
            else if (family == "binomial") {
                loss_func = auc;
            }
        }
        else if (user_loss == "deviance")
        {
            if (family == "gaussian"){
                loss_func = mean_squared_error;
            }
            else if (family == "binomial") {
                loss_func = deviance_binomial;
            }
        }
        else {
            loss_func = lossMap[user_loss];
        }
        return loss_func;
    }

    static double mean_squared_error(const Eigen::Ref<const Eigen::VectorXd> & actual,
                                     const Eigen::Ref<const Eigen::VectorXd> & predicted,
                                     const Eigen::Ref<const Eigen::VectorXi> & test_idx) {
        double error = 0.0;
        for (int i = 0; i < test_idx.size(); ++i) {
            error += std::pow(actual[test_idx[i]] - predicted[test_idx[i]], 2) / test_idx.size();
        }
        return error;
    }

    static double mean_absolute_error(const Eigen::Ref<const Eigen::VectorXd> & actual,
                                      const Eigen::Ref<const Eigen::VectorXd> & predicted,
                                      const Eigen::Ref<const Eigen::VectorXi> & test_idx) {
        double error = 0.0;
        for (int i = 0; i < test_idx.size(); ++i) {
            error += std::abs(actual[test_idx[i]] - predicted[test_idx[i]]) / test_idx.size();
        }
        return error;
    }

    static double auc(const Eigen::Ref<const Eigen::VectorXd> & actual,
                      const Eigen::Ref<const Eigen::VectorXd> & predicted,
                      const Eigen::Ref<const Eigen::VectorXi> & test_idx) {

        // subset to test y / yhat
        const int test_size = test_idx.size();
        Eigen::VectorXd actual_sub(test_size);
        Eigen::VectorXd pred_sub(test_size);
        for (int i = 0; i < test_size; ++i) {
            actual_sub[i] = actual[test_idx[i]];
            pred_sub[i] = predicted[test_idx[i]];
        }

        // get rankings
        const int n = pred_sub.size();
        std::vector<size_t> indx(n);
        std::iota(indx.begin(), indx.end(), 0);
        std::sort(indx.begin(), indx.end(), [&pred_sub](int i1, int i2) {
            return pred_sub[i1] < pred_sub[i2];
        });

        // compute mann whitney u --> use to get auc
        int n1 = 0;
        double rank_sum = 0;
        for (int i = 0; i < n; ++i) {
            if (actual_sub[indx[i]] == 1) {
                ++n1;
                rank_sum += i + 1;
            }
        }
        double u_value = rank_sum - (n1 * (n1 + 1)) / 2.0;
        return u_value / (n1 * (n - n1));
    }

    static double deviance_binomial(const Eigen::Ref<const Eigen::VectorXd> & actual,
                                    const Eigen::Ref<const Eigen::VectorXd> & predicted,
                                    const Eigen::Ref<const Eigen::VectorXi> & test_idx) {
        double error = 0.0;
        for (int i = 0; i < test_idx.size(); ++i){
            error += (actual[test_idx[i]] * predicted[test_idx[i]] - log(1.0 + exp(predicted[test_idx[i]]))) / test_idx.size();
        }
        return -2 * error;
    }
};

#endif // XRNET_CV_H




