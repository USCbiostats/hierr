#' Define model for right-censored time-to-event data.
#'
#' @description Defines the model we are going to use to model the right-censored outcome.
#'
#' @param model String. Model name. Currently \code{cox} is supported.
#' @export
define_surv <- function(model = "cox") {
    # Checks
    # More will be added later
    surv_obj <- list(
        model = model
    )
    return(surv_obj)
}

