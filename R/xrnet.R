#' @useDynLib xrnet, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom stats predict
#' @importFrom bigmemory is.big.matrix
#' @importFrom methods is
NULL

#' Fit hierarchical regularized regression model
#'
#' @description Fits hierarchical regularized regression model that enables the incorporation of external data
#' for predictor variables. Both the predictor variables and external data can be regularized
#' by the most common penalties (lasso, ridge, elastic net).
#' Solutions are computed across a two-dimensional grid of penalties (a separate penalty path is computed
#' for the predictors and external variables). Currently support regularized linear and logistic regression,
#' future extensions to other outcomes (i.e. Cox regression) will be implemented in the next major update.
#'
#' @param x predictor design matrix of dimension \eqn{n x p}, matrix options include:
#' \itemize{
#'    \item matrix
#'    \item big.matrix
#'    \item filebacked.big.matrix
#'    \item sparse matrix (dgCMatrix)
#' }
#' @param y outcome vector of length \eqn{n}
#' @param external (optional) external data design matrix of dimension \eqn{p x q}, matrix options include:
#' \itemize{
#'     \item matrix
#'     \item sparse matrix (dgCMatrix)
#' }
#' @param unpen (optional) unpenalized predictor design matrix, matrix options include:
#' \itemize{
#'     \item matrix
#' }
#' @param family error distribution for outcome variable, options include:
#' \itemize{
#'     \item gaussian
#'     \item binomial
#' }
#' @param penalty_main specifies regularization object for x. See \code{\link{define_penalty}} for more details.
#' @param penalty_external specifies regularization object for external. See \code{\link{define_penalty}} for more details.
#' @param weights optional vector of observation-specific weights. Default is 1 for all observations.
#' @param standardize indicates whether x and/or external should be standardized. Default is c(TRUE, TRUE).
#' @param intercept indicates whether an intercept term is included for x and/or external.
#' Default is c(TRUE, FALSE).
#' @param control specifies xrnet control object. See \code{\link{xrnet.control}} for more details.
#'
#' @details This function extends the coordinate descent algorithm of the R package \code{glmnet} to allow the
#' type of regularization (i.e. ridge, lasso) to be feature-specific. In addition, elements of the R package \code{biglasso}
#' are utilized to enable the use of standard R matrices, memory-mapped matrices from the \code{bigmemory} package,
#' or sparse matrices from the \code{Matrix} package.
#'
#' @references
#' Jerome Friedman, Trevor Hastie, Robert Tibshirani (2010).
#' Regularization Paths for Generalized Linear Models via Coordinate Descent.
#' Journal of Statistical Software, 33(1), 1-22. URL http://www.jstatsoft.org/v33/i01/.
#'
#' @references
#' Zeng, Y., and Breheny, P. (2017).
#' The biglasso Package: A Memory- and Computation-Efficient Solver for Lasso Model Fitting with Big Data in R.
#' arXiv preprint arXiv:1701.05936. URL https://arxiv.org/abs/1701.05936.
#'
#' @references
#' Michael J. Kane, John Emerson, Stephen Weston (2013).
#' Scalable Strategies for Computing with Massive Data.
#' Journal of Statistical Software, 55(14), 1-19. URL http://www.jstatsoft.org/v55/i14/.
#'
#' @return A list of class \code{xrnet} with components:
#' \item{beta0}{matrix of first-level intercepts indexed by penalty values}
#' \item{betas}{3-dimensional array of first-level penalized coefficients indexed by penalty values}
#' \item{gammas}{3-dimensional array of first-level non-penalized coefficients indexed by penalty values}
#' \item{alpha0}{matrix of second-level intercepts indexed by penalty values}
#' \item{alphas}{3-dimensional array of second-level external data coefficients indexed by penalty values}
#' \item{penalty}{vector of first-level penalty values}
#' \item{penalty_ext}{vector of second-level penalty values}
#' \item{family}{error distribution for outcome variable}
#' \item{num_passes}{total number of passes over the data in the coordinate descent algorithm}
#' \item{status}{error status for xrnet fitting}
#' \itemize{
#'     \item 0 = OK
#'     \item 1 = Error/Warning
#' }
#' \item{error_msg}{description of error}
#'
#' @examples
#' ### hierarchical regularized linear regression ###
#' data(GaussianExample)
#'
#' ## define penalty for predictors and external variables
#' ## default is ridge for predictors and lasso for external
#' ## see define_penalty() function for more details
#'
#' penMain <- define_penalty(0, num_penalty = 20)
#' penExt <- define_penalty(1, num_penalty = 20)
#'
#' ## fit model with defined regularization
#' fit_xrnet <- xrnet(
#'     x = x_linear,
#'     y = y_linear,
#'     external = ext_linear,
#'     family = "gaussian",
#'     penalty_main = penMain,
#'     penalty_external = penExt
#' )

#' @export
xrnet <- function(x,
                  y,
                  external = NULL,
                  unpen = NULL,
                  family = c("gaussian", "binomial", "survival"),
                  penalty_main = define_penalty(),
                  penalty_external = define_penalty(),
                  survival_family = define_surv(),
                  weights = NULL,
                  standardize = c(TRUE, TRUE),
                  intercept = c(TRUE, FALSE),
                  control = list())
{

    # function call
    this.call <- match.call()

    # check error distribution for y
    family <- match.arg(family)

    ## Prepare x and y ##

    # check type of x matrix
    if (is(x, "matrix")) {
        if (typeof(x) != "double")
            stop("Error: x must be of type double")
        mattype_x <- 1
    }
    else if (is.big.matrix(x)) {
        if (bigmemory::describe(x)@description$type != "double")
            stop("Error: x must be of type double")
        mattype_x <- 2
    } else if ("dgCMatrix" %in% class(x)) {
        if (typeof(x@x) != "double")
            stop("Error: x must be of type double")
        mattype_x <- 3
    } else {
        stop("Error: x must be a standard R matrix, big.matrix, filebacked.big.matrix, or dgCMatrix")
    }


    # ESK: Checks on survival type:
    if (class(y) != "Surv" & family == "survival") {
       stop("Models with 'family = survival' must have a 'Surv' object outcome.")
    }

    # check type of y
    y <- as.double(drop(y))

    # check dimensions of x and y
    nr_x <- NROW(x)
    nc_x <- NCOL(x)
    y_len <- NROW(y)

    if (y_len != nr_x) {
        stop(
            paste(
                "Error: Length of y (", y_len,
                ") not equal to the number of rows of x (", nr_x,")",
                sep = ""
             )
        )
    }

    ## Prepare external ##
    is_sparse_ext = FALSE
    if (!is.null(external)) {

        # check if external is a sparse matrix
        if (is(external, "sparseMatrix")) {
            is_sparse_ext = TRUE
        } else {
            # convert to matrix
            if (class(external) != "matrix") {
                external <- as.matrix(external)
            }
            if (typeof(external) != "double") {
                stop("Error: external must be of type double")
            }
        }

        # check dimensions
        nr_ext <- NROW(external)
        nc_ext <- NCOL(external)

        if (nc_x != nr_ext) {
            stop(
                paste("Error: Number of columns in x (", nc_x,
                      ") not equal to the number of rows in external (", nr_ext,
                      ")", sep = ""
                )
            )
        }

    } else {
        external <- matrix(vector("numeric", 0), 0, 0)
        nr_ext <- as.integer(0)
        nc_ext <- as.integer(0)
    }

    ## Prepare unpenalized covariates ##
    if (!is.null(unpen)) {

        # check dimensions
        nc_unpen <- NCOL(unpen)

        if (y_len != NROW(unpen)) {
            stop(
                paste(
                    "Error: Length of y (", y_len,
                    ") not equal to the number of rows of unpen (", NROW(unpen),
                    ")", sep = ""
                )
            )
        }

        # convert unpen to matrix
        if (class(unpen) != "matrix") {
            unpen <- as.matrix(unpen)
        }
        if (typeof(x) != "double") {
            stop("Error: unpen contains non-numeric values")
        }
    } else {
        unpen <- matrix(vector("numeric", 0), 0, 0)
        nc_unpen <- as.integer(0)
    }

    # set weights
    if (is.null(weights)) {
        weights <- as.double(rep(1, nr_x))
    } else if (length(weights) != y_len) {
        stop(
            paste(
                "Error: Length of weights (", length(weights),
                ") not equal to length of y (", y_len,
                ")", sep = ""
            )
        )
    } else if (any(weights < 0)) {
        stop("Error: weights can only contain non-negative values")
    } else {
        weights <- as.double(weights)
    }

    # check penalty objects
    penalty <- initialize_penalty(
        penalty_main = penalty_main,
        penalty_external = penalty_external,
        nr_x = nr_x,
        nc_x = nc_x,
        nc_unpen = nc_unpen,
        nr_ext = nr_ext,
        nc_ext = nc_ext,
        intercept = intercept
    )

    # check control object
    control <- do.call("xrnet.control", control)
    control <- initialize_control(
        control_obj = control,
        nc_x = nc_x,
        nc_unpen = nc_unpen,
        nc_ext = nc_ext,
        intercept = intercept
    )

    # ESK: To incorporate survival model correctly.
    if (family == "survival") {
        family = survival_family$model
    }

    # fit model
    fit <- fitModelRcpp(
        x = x,
        mattype_x = mattype_x,
        y = y,
        ext = external,
        is_sparse_ext = is_sparse_ext,
        fixed = unpen,
        weights_user = weights,
        intr = intercept,
        stnd = standardize,
        penalty_type = penalty$ptype,
        cmult = penalty$cmult,
        quantiles = c(penalty$quantile, penalty$quantile_ext),
        num_penalty = c(penalty$num_penalty, penalty$num_penalty_ext),
        penalty_ratio = c(penalty$penalty_ratio, penalty$penalty_ratio_ext),
        penalty_user = penalty$user_penalty,
        penalty_user_ext = penalty$user_penalty_ext,
        lower_cl = control$lower_limits,
        upper_cl = control$upper_limits,
        family = family,
        thresh = control$tolerance,
        maxit = control$max_iterations,
        ne = control$dfmax,
        nx = control$pmax
    )

    # check status of model fit
    if (fit$status == 0) {
        fit$status <- "0 (OK)"

        # Create arrays ordering coefficients by 1st level penalty / 2nd level penalty
        fit$beta0 <- matrix(
            fit$beta0,
            nrow = penalty$num_penalty,
            ncol = penalty$num_penalty_ext,
            byrow = TRUE
        )

        dim(fit$betas) <- c(nc_x, penalty$num_penalty_ext, penalty$num_penalty)
        fit$betas <- aperm(fit$betas, c(1, 3, 2))

        if (intercept[2]) {
            fit$alpha0 <- matrix(
                fit$alpha0,
                nrow = penalty$num_penalty,
                ncol = penalty$num_penalty_ext, byrow = TRUE
            )
        } else {
            fit$alpha0 <- NULL
        }

        if (nc_ext > 0) {
            dim(fit$alphas) <- c(nc_ext, penalty$num_penalty_ext, penalty$num_penalty)
            fit$alphas <- aperm(fit$alphas, c(1, 3, 2))
        } else {
            fit$alphas <- NULL
            fit$penalty_ext <- NULL
        }

        if (nc_unpen > 0) {
            dim(fit$gammas) <- c(nc_unpen, penalty$num_penalty_ext, penalty$num_penalty)
            fit$gammas <- aperm(fit$gammas, c(1, 3, 2))
        } else {
            fit$gammas <- NULL
        }

        # Modify gradient here.

    } else {
        if (fit$status == 1) {
            fit$error_msg <- "Max number of iterations reached"
            warning("Max number of iterations reached")
        }
        fit$status <- "1 (Error/Warning)"
    }

    fit$call <- this.call
    class(fit) <- "xrnet"
    return(fit)
}

#' Control function for xrnet fitting
#'
#' @description Control function for \code{\link{xrnet}} fitting.
#'
#' @param tolerance positive convergence criterion. Default is 1e-08.
#' @param max_iterations maximum number of iterations to run coordinate gradient descent
#' across all penalties before returning an error. Default is 1e+05.
#' @param dfmax maximum number of variables allowed in model. Default
#' is \eqn{ncol(x) + ncol(external) + intercept[1] + intercept[2]}.
#' @param pmax maximum number of variables with nonzero coefficient estimate.
#' Default is \eqn{min(2 * dfmax + 20, ncol(x) + ncol(external) + intercept[2])}.
#' @param lower_limits vector of lower limits for each coefficient. Default is -Inf for all variables.
#' @param upper_limits vector of upper limits for each coefficient. Default is Inf for all variables.

#' @export
xrnet.control <- function(tolerance = 1e-08,
                          max_iterations = 1e+05,
                          dfmax = NULL,
                          pmax = NULL,
                          lower_limits = NULL,
                          upper_limits = NULL) {

    if (tolerance <= 0) {
        stop("Error: tolerance must be greater than 0")
    }

    if (max_iterations <= 0 || as.integer(max_iterations) != max_iterations) {
        stop("Error: max_iterations must be a positive integer")
    }

    control_obj <- list(
        tolerance = tolerance,
        max_iterations = max_iterations,
        dfmax = dfmax,
        pmax = pmax,
        lower_limits = lower_limits,
        upper_limits = upper_limits
    )
}


initialize_penalty <- function(penalty_main,
                               penalty_external,
                               nr_x,
                               nc_x,
                               nc_unpen,
                               nr_ext,
                               nc_ext,
                               intercept) {

    names(penalty_external) <- c(
        "penalty_type_ext",
        "quantile_ext",
        "num_penalty_ext",
        "penalty_ratio_ext",
        "user_penalty_ext",
        "custom_multiplier_ext"
    )

    penalty_obj <- c(penalty_main, penalty_external)

    # check penalty object for x
    if (length(penalty_obj$penalty_type) > 1) {
        if (length(penalty_obj$penalty_type) != nc_x) {
            stop(
                "Error: Length of penalty_type (",
                length(penalty_obj$penalty_type),
                ") not equal to number of columns in x (",
                nc_x,")"
            )
        }
    } else {
        penalty_obj$penalty_type <- rep(penalty_obj$penalty_type, nc_x)
    }

    if (is.null(penalty_obj$penalty_ratio)) {
        if (penalty_obj$user_penalty[1] == 0) {
            if (nr_x > nc_x) {
                penalty_obj$penalty_ratio <- 1e-04
            } else {
                penalty_obj$penalty_ratio <- 0.01
            }
            if (penalty_obj$num_penalty < 3) {
                penalty_obj$num_penalty <- 3
                stop("Warning: num_penalty must be at least 3
                     when automatically computing penalty path")
            }
        } else {
            penalty_obj$user_penalty <- rev(sort(penalty_obj$user_penalty))
            penalty_obj$penalty_ratio <- 0.0
        }
    }

    if (is.null(penalty_obj$custom_multiplier)) {
        penalty_obj$custom_multiplier <- rep(1.0, nc_x)
    } else if (length(penalty_obj$custom_multiplier) != nc_x) {
        stop(
            "Error: Length of custom_multiplier (",
            length(penalty_obj$custom_multiplier),
            ") not equal to number of columns in x (",
            nc_x, ")"
        )
    }

    # check penalty object for external
    if (nc_ext > 0) {
        if (length(penalty_obj$penalty_type_ext) > 1) {
            if (length(penalty_obj$penalty_type_ext) != nc_ext) {
                stop(
                    "Error: Length of penalty_type_ext (",
                    length(penalty_obj$penalty_type_ext),
                    ") not equal to number of columns in external (",
                    nc_ext,
                    ")"
                )
            }
        } else {
            penalty_obj$penalty_type_ext <- rep(penalty_obj$penalty_type_ext, nc_ext)
        }

        if (is.null(penalty_obj$penalty_ratio_ext)) {
            if (penalty_obj$user_penalty_ext[1] == 0) {
                if (nr_ext > nc_ext) {
                    penalty_obj$penalty_ratio_ext <- 1e-04
                } else {
                    penalty_obj$penalty_ratio_ext <- 0.01
                }
                if (penalty_obj$num_penalty_ext < 3) {
                    penalty_obj$num_penalty_ext <- 3
                    stop("Warning: num_penalty_ext must be at least
                         3 when automatically computing penalty path")
                }
            } else {
                penalty_obj$user_penalty_ext <- rev(sort(penalty_obj$user_penalty_ext))
                penalty_obj$penalty_ratio_ext <- 0.0
            }
        }

        if (is.null(penalty_obj$custom_multiplier_ext)) {
            penalty_obj$custom_multiplier_ext <- rep(1.0, nc_ext)
        } else if (length(penalty_obj$custom_multiplier_ext) != nc_ext && nc_ext > 0) {
            stop(
                "Error: Length of custom_multiplier_ext (",
                length(penalty_obj$custom_multiplier_ext),
                ") not equal to number of columns in external (",
                nc_ext, ")"
            )
        }
    } else {
        penalty_obj$penalty_type_ext <- NULL
        penalty_obj$num_penalty_ext <- 1
        penalty_obj$penalty_ratio_ext <- 0
        penalty_obj$custom_multiplier_ext <- numeric(0)
    }

    # vectors holding penalty type and multipliers across all variables
    if (intercept[2]) {
        penalty_obj$ptype <- c(
            penalty_obj$penalty_type,
            rep(0.0, nc_unpen),
            0.0,
            penalty_obj$penalty_type_ext
        )
        penalty_obj$cmult <- c(
            penalty_obj$custom_multiplier,
            rep(0.0, nc_unpen),
            0.0,
            penalty_obj$custom_multiplier_ext
        )
    } else {
        penalty_obj$ptype <- c(
            penalty_obj$penalty_type,
            rep(0.0, nc_unpen),
            penalty_obj$penalty_type_ext
        )
        penalty_obj$cmult <- c(
            penalty_obj$custom_multiplier,
            rep(0.0, nc_unpen),
            penalty_obj$custom_multiplier_ext
        )
    }
    return(penalty_obj)
}

initialize_control <- function(control_obj,
                               nc_x,
                               nc_unpen,
                               nc_ext,
                               intercept) {

    if (is.null(control_obj$dfmax)) {
        control_obj$dfmax <- as.integer(nc_x + nc_ext + nc_unpen + intercept[1] + intercept[2])
    } else if (control_obj$dfmax <= 0 || as.integer(control_obj$dfmax) != control_obj$dfmax) {
        stop("Error: dfmax can only contain postive integers")
    }

    if (is.null(control_obj$pmax)) {
        control_obj$pmax <- as.integer(min(2 * control_obj$dfmax + 20, nc_x + nc_ext + nc_unpen + intercept[2]))
    } else if (control_obj$pmax <= 0 || as.integer(control_obj$pmax) != control_obj$pmax) {
        stop("Error: pmax can only contain positive integers")
    }

    if (is.null(control_obj$lower_limits)) {
        control_obj$lower_limits <- rep(-Inf, nc_x + nc_ext + nc_unpen + intercept[2])
    } else if (length(control_obj$lower_limits) != nc_x + nc_ext + nc_unpen) {
        stop(
            "Error: Length of lower_limits (",
            length(control_obj$lower_limits),
            ") not equal to sum of number of columns in x, unpen, and external (",
            nc_x + nc_ext + nc_unpen, ")"
        )
    } else if (intercept[2]) {
        control_obj$lower_limits <- c(
            control_obj$lower_limits[1:(nc_x + nc_unpen)],
            -Inf,
            control_obj$lower_limits[(nc_x + nc_unpen + 1):length(control_obj$lower_limits)]
        )
    }

    if (is.null(control_obj$upper_limits)) {
        control_obj$upper_limits <- rep(Inf, nc_x + nc_ext + nc_unpen + intercept[2])
    } else if (length(control_obj$upper_limits) != nc_x + nc_ext + nc_unpen) {
        stop(
            "Error: Length of upper_limits (",
            length(control_obj$upper_limits),
            ") not equal to sum of number of columns in x, unpen, and external (",
            nc_x + nc_ext + nc_unpen, ")"
        )
    } else if (intercept[2]) {
        control_obj$upper_limits <- c(
            control_obj$upper_limits[1:(nc_x + nc_unpen)],
            -Inf,
            control_obj$upper_limits[(nc_x + nc_unpen + 1):length(control_obj$upper_limits)]
        )
    }
    return(control_obj)
}
