#include <string.h>
#include "DataFunctions.h"
#include "XrnetCV.h"
#include "XrnetUtils.h"
#include "CoordDescTypes.h"
#include "GaussianSolver.h"
#include "BinomialSolver.h"

template <typename TX, typename TZ>
Eigen::VectorXd fitModelCV(const TX & x,
                           const bool & is_sparse_x,
                           const Eigen::Ref<const Eigen::MatrixXd> & y,
                           const TZ & ext,
                           const Eigen::Ref<const Eigen::MatrixXd> & fixed,
                           Eigen::VectorXd weights_user,
                           const Rcpp::LogicalVector & intr,
                           const Rcpp::LogicalVector & stnd,
                           const Eigen::Ref<const Eigen::VectorXd> & penalty_type,
                           const Eigen::Ref<const Eigen::VectorXd> & cmult,
                           const Eigen::Ref<const Eigen::VectorXd> & quantiles,
                           const Rcpp::IntegerVector & num_penalty,
                           const Rcpp::NumericVector & penalty_ratio,
                           const Eigen::Ref<const Eigen::VectorXd> & penalty_user,
                           const Eigen::Ref<const Eigen::VectorXd> & penalty_user_ext,
                           Eigen::VectorXd lower_cl,
                           Eigen::VectorXd upper_cl,
                           const std::string & family,
                           const std::string & user_loss,
                           const Eigen::Ref<const Eigen::VectorXi> & test_idx,
                           const double & thresh,
                           const int & maxit,
                           const int & ne,
                           const int & nx) {

    // initialize objects to hold means, variances, sds of all variables
    const int n = x.rows();
    const int nv_x = x.cols();
    const int nv_fixed = fixed.size() == 0 ? 0 : fixed.cols();
    const int nv_ext = ext.size() == 0 ? 0 : ext.cols();
    const int nv_total = nv_x + nv_fixed + intr[1] + nv_ext;
    Eigen::VectorXd xm = Eigen::VectorXd::Constant(nv_total, 0.0);
    Eigen::VectorXd cent = Eigen::VectorXd::Constant(nv_total, 0.0);
    Eigen::VectorXd xv = Eigen::VectorXd::Constant(nv_total, 1.0);
    Eigen::VectorXd xs = Eigen::VectorXd::Constant(nv_total, 1.0);

    // map to correct size of fixed matrix
    Eigen::Map<const Eigen::MatrixXd> fixedmap(fixed.data(), fixed.rows(), nv_fixed);

    // scale user weights
    weights_user.array() = weights_user.array() / weights_user.sum();

    // compute moments of matrices and create XZ (if external data present)
    const bool center_x = intr[0] && !is_sparse_x;
    compute_moments(x, weights_user, xm, cent, xv, xs, center_x, stnd[0], 0);
    compute_moments(fixedmap, weights_user, xm, cent, xv, xs, center_x, stnd[0], nv_x);
    const Eigen::MatrixXd xz = create_XZ(
        x, ext, xm, cent, weights_user,
        xv, xs, intr[1], stnd[1], nv_x + nv_fixed
    );

    // choose solver based on outcome
    std::unique_ptr<CoordSolver<TX> > solver;
    if (family == "gaussian") {
        solver.reset(
            new GaussianSolver<TX>(
                    y, x, fixedmap, xz, cent.data(), xv.data(), xs.data(),
                    weights_user, intr[0], penalty_type.data(),
                    cmult.data(), quantiles, upper_cl.data(),
                    lower_cl.data(), ne, nx, thresh, maxit
            )
        );

    }
    else if (family == "binomial") {
        solver.reset(
            new BinomialSolver<TX>(
                    y, x, fixedmap, xz, cent.data(), xv.data(),
                    xs.data(), weights_user, intr[0], penalty_type.data(),
                    cmult.data(), quantiles, upper_cl.data(),
                    lower_cl.data(), ne, nx, thresh, maxit
            )
        );
    }
    // Object to hold results for all penalty combinations
    const int num_combn = num_penalty[0] * num_penalty[1];
    XrnetCV<TX, TZ> results = XrnetCV<TX, TZ>(
        n, nv_x, nv_fixed, nv_ext, nv_total,
        intr[0], intr[1], ext, xm.data(), cent.data(),
        xs.data(), solver->getYm(), solver->getYs(), num_combn, family, user_loss,
        test_idx, x, y
    );

    // compute penalty path for 1st level variables
    Eigen::VectorXd path(num_penalty[0]);

    compute_penalty(
        path, penalty_user, penalty_type[0],
        penalty_ratio[0], solver->getGradient(),
        solver->getCmult(), 0, nv_x, solver->getYs()
    );

    // compute penalty path for 2nd level variables
    Eigen::VectorXd path_ext(num_penalty[1]);
    if (nv_ext > 0) {
        compute_penalty(
            path_ext, penalty_user_ext,
            penalty_type[nv_x + nv_fixed + intr[1]],
            penalty_ratio[1], solver->getGradient(),
            solver->getCmult(), nv_x + nv_fixed + intr[1],
            nv_total, solver->getYs()
        );
    } else {
        path_ext[0] = 0.0;
    }

    // solve grid of penalties in decreasing order
    double b0_outer = solver->getBeta0();
    double dev_outer = solver->getDeviance();
    Eigen::VectorXd betas_outer = solver->getBetas();

    int idx_pen = 0;
    for (int m = 0; m < num_penalty[0]; ++m) {
        solver->setPenalty(path[m], 0);
        for (int m2 = 0; m2 < num_penalty[1]; ++m2, ++idx_pen) {
            solver->setPenalty(path_ext[m2], 1);
            if (m2 == 0 && num_penalty[1] > 1) {
                solver->warm_start(b0_outer, betas_outer);
                solver->update_strong(path, path_ext, m, m2);
                solver->solve();
                b0_outer = solver->getBeta0();
                betas_outer = solver->getBetas();
                dev_outer = solver->getDeviance();
            }
            else {
                solver->update_strong(path, path_ext, m, m2);
                solver->solve();
            }
            results.add_results(solver->getBeta0(), solver->getBetas(), solver->getDeviance(), idx_pen);
        }
    }

    // return results
    return results.get_error_mat();
}


// [[Rcpp::export]]
Eigen::VectorXd fitModelCVRcpp(SEXP x,
                               const int mattype_x,
                               const Eigen::Map<Eigen::MatrixXd> y,
                               SEXP ext,
                               const bool & is_sparse_ext,
                               const Eigen::Map<Eigen::MatrixXd> fixed,
                               Eigen::VectorXd weights_user,
                               const Rcpp::LogicalVector & intr,
                               const Rcpp::LogicalVector & stnd,
                               const Eigen::Map<Eigen::VectorXd> penalty_type,
                               const Eigen::Map<Eigen::VectorXd> cmult,
                               const Eigen::Map<Eigen::VectorXd> quantiles,
                               const Rcpp::IntegerVector & num_penalty,
                               const Rcpp::NumericVector & penalty_ratio,
                               const Eigen::Map<Eigen::VectorXd> penalty_user,
                               const Eigen::Map<Eigen::VectorXd> penalty_user_ext,
                               Eigen::VectorXd lower_cl,
                               Eigen::VectorXd upper_cl,
                               const std::string & family,
                               const std::string & user_loss,
                               const Eigen::Map<Eigen::VectorXi> test_idx,
                               const double & thresh,
                               const int & maxit,
                               const int & ne,
                               const int & nx) {

    if (mattype_x == 1) {
        const bool is_sparse_x = false;
        Rcpp::NumericMatrix x_mat(x);
        MapMat xmap((const double *) &x_mat[0], x_mat.rows(), x_mat.cols());
        if (is_sparse_ext)
            return fitModelCV<MapMat, MapSpMat>(
                    xmap, is_sparse_x, y, Rcpp::as<MapSpMat>(ext),
                    fixed, weights_user, intr, stnd, penalty_type,
                    cmult, quantiles, num_penalty, penalty_ratio,
                    penalty_user, penalty_user_ext, lower_cl,
                    upper_cl, family, user_loss, test_idx, thresh,
                    maxit, ne, nx
                );
        else {
            Rcpp::NumericMatrix ext_mat(ext);
            MapMat extmap((const double *) &ext_mat[0], ext_mat.rows(), ext_mat.cols());
            return fitModelCV<MapMat, MapMat>(
                    xmap, is_sparse_x, y, extmap, fixed,
                    weights_user, intr, stnd, penalty_type,
                    cmult, quantiles, num_penalty, penalty_ratio,
                    penalty_user, penalty_user_ext, lower_cl,
                    upper_cl, family, user_loss, test_idx,
                    thresh, maxit, ne, nx
                );
        }
    } else if (mattype_x == 2) {
        const bool is_sparse_x = false;
        Rcpp::S4 x_info(x);
        Rcpp::XPtr<BigMatrix> xptr((SEXP) x_info.slot("address"));
        MapMat xmap((const double *)xptr->matrix(), xptr->nrow(), xptr->ncol());
        if (is_sparse_ext) {
            return fitModelCV<MapMat, MapSpMat>(
                    xmap, is_sparse_x, y, Rcpp::as<MapSpMat>(ext), fixed,
                    weights_user, intr, stnd, penalty_type,
                    cmult, quantiles, num_penalty, penalty_ratio,
                    penalty_user, penalty_user_ext, lower_cl,
                    upper_cl, family, user_loss, test_idx,
                    thresh, maxit, ne, nx
                );
        }
        else {
            Rcpp::NumericMatrix ext_mat(ext);
            MapMat extmap((const double *) &ext_mat[0], ext_mat.rows(), ext_mat.cols());
            return fitModelCV<MapMat, MapMat>(
                    xmap, is_sparse_x, y, extmap, fixed, weights_user,
                    intr, stnd, penalty_type, cmult,
                    quantiles, num_penalty, penalty_ratio,
                    penalty_user, penalty_user_ext, lower_cl,
                    upper_cl, family, user_loss, test_idx,
                    thresh, maxit, ne, nx
                );
        }
    } else {
        const bool is_sparse_x = true;
        if (is_sparse_ext) {
            return fitModelCV<MapSpMat, MapSpMat>(
                    Rcpp::as<MapSpMat>(x), is_sparse_x, y, Rcpp::as<MapSpMat>(ext), fixed,
                    weights_user, intr, stnd, penalty_type,
                    cmult, quantiles, num_penalty, penalty_ratio,
                    penalty_user, penalty_user_ext, lower_cl,
                    upper_cl, family, user_loss, test_idx,
                    thresh, maxit, ne, nx
            );
        }
        else {
            Rcpp::NumericMatrix ext_mat(ext);
            MapMat extmap((const double *) &ext_mat[0], ext_mat.rows(), ext_mat.cols());
            return fitModelCV<MapSpMat, MapMat>(
                    Rcpp::as<MapSpMat>(x), is_sparse_x, y, extmap,
                    fixed, weights_user, intr, stnd, penalty_type, cmult,
                    quantiles, num_penalty, penalty_ratio,
                    penalty_user, penalty_user_ext, lower_cl,
                    upper_cl, family, user_loss, test_idx,
                    thresh, maxit, ne, nx
            );
        }
    }
}
