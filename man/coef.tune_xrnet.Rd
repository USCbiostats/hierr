% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/coef_tune_xrnet.R
\name{coef.tune_xrnet}
\alias{coef.tune_xrnet}
\title{Get coefficient estimates from "tune_xrnet" model object.}
\usage{
\method{coef}{tune_xrnet}(object, p = "opt", pext = "opt", ...)
}
\arguments{
\item{object}{A \code{\link{tune_xrnet}} object.}

\item{p}{vector of penalty values to apply to predictor variables.
Default is optimal value in tune_xrnet object.}

\item{pext}{vector of penalty values to apply to external data variables.
Default is optimal value in tune_xrnet object.}

\item{...}{pass other arguments to xrnet function (if needed).}
}
\value{
A list with coefficient estimates at each of the requested penalty
combinations.
\item{beta0}{matrix of first-level intercepts indexed by penalty values, NULL
if no first-level intercept in original model fit.}
\item{betas}{3-dimensional array of first-level penalized coefficients
indexed by penalty values.}
\item{gammas}{3-dimensional array of first-level non-penalized coefficients
indexed by penalty values, NULL if unpen NULL in original model fit.}
\item{alpha0}{matrix of second-level intercepts indexed by penalty values,
NULL if no second-level intercept in original model fit.}
\item{alphas}{3-dimensional array of second-level external data coefficients
indexed by penalty values, NULL if external NULL in original model fit.}
}
\description{
Returns coefficients from 'xrnet' model. Note that we currently
only support returning coefficient estimates that are in the original
path(s).
}
\examples{
## Cross validation of hierarchical linear regression model
data(GaussianExample)

## 5-fold cross validation
cv_xrnet <- tune_xrnet(
  x = x_linear,
  y = y_linear,
  external = ext_linear,
  family = "gaussian",
  control = xrnet_control(tolerance = 1e-6)
)

## Get coefficient estimates at optimal penalty combination
coef_opt <- coef(cv_xrnet)
}
