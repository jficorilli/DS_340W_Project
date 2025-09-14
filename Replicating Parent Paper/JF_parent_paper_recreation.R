# Jason Ficorilli - Recreating Parent Paper
# Adapted/consolidated from three R code files in first author's GitHub: MEPS.R, twostageSL.R, and Estimators.R. 

# Loading relevant libraries

# For parallel processing:
library(parallel)
library(doParallel)
library(foreach)

# Used in MEPS.R
library(mgcv)
library(quadprog)
library(SuperLearner)
library(earth)
library(pscl) 
library(VGAM) 
library(cplm)
library(caret) 
library(randomForest) 
library(e1071) 
library(gbm) 
library(moments) 
library(flexsurv) 
library(survival) 
library(quantreg) 
library(sandwich)
library(glmnet)
library(foreign)
library(survey)
library(dplyr)
library(ggplot2)


# Code to load estimator functions, as used in Estimators.R:

##################################### One-part model #######################################################

#==================================================#
#	Zero-Inflated Poisson Model
#==================================================#

SL.zip <- function(Y, X, newX, family, obsWeights, ...){
  if(family$family=="gaussian"){
    # round outcome Y to be interger
    Y.int <- round(Y)
    suppressWarnings(
      fit.zip <- zeroinfl(Y.int ~ . | ., data=X,weights = obsWeights)
    )
    pred <- predict(fit.zip, newdata=newX, type="response")
    fit <- list(object = fit.zip)
    class(fit) <- "SL.glm" # can use predict.SL.glm
    out <- list(pred=pred, fit=fit)
    return(out)
  }else{
    stop("SL.zip not written for binomial family")
  }
}

#==================================================#
# Zero-Inflated Negative Binomial Model
#==================================================#

SL.zinb <- function(Y, X, newX, family, obsWeights, ...){
  if(family$family=="gaussian"){
    # round outcome Y to be interger
    Y.int <- round(Y)
    suppressWarnings(
      fit.zinb <- zeroinfl(Y.int ~ . | ., data=X,weights = obsWeights,dist = "negbin")
    )
    pred <- predict(fit.zinb, newdata=newX, type="response")
    fit <- list(object = fit.zinb)
    class(fit) <- "SL.glm" # can use predict.SL.glm
    out <- list(pred=pred, fit=fit)
    return(out)
  }else{
    stop("SL.zinb not written for binomial family")
  }
}

#==================================================#
#	Tobit Model
#==================================================#

SL.tobit <- function(Y, X, newX, family, obsWeights, ...){
  if(family$family=="gaussian"){
    suppressWarnings(
      fit.tobit <- vglm(Y ~., tobit(Lower = 0,type.fitted = "censored"),data=X,maxit=100)
    )
    pred <- predict(fit.tobit, newdata=newX, type="response")
    # in case generate negative prediction
    pred[pred<0]=0
    fit <- list(object = fit.tobit)
    class(fit) <- "SL.tobit"
    out <- list(pred=pred, fit=fit)
    return(out)
  }else{
    stop("SL.tobit not written for binomial family")
  }
}

predict.SL.tobit <- function(object, newdata, ...) {
  # newdata must be a dataframe, not a matrix.
  if (is.matrix(newdata)) {
    newdata = as.data.frame(newdata)
  }
  pred <- predict(object = object$object, newdata = newdata, type = "response")
  # in case generate negative prediction
  pred[pred<0]=0
  pred
}

#==================================================#
#	Tweedie Model
#==================================================#

SL.tweedie <- function(Y, X, newX, family, obsWeights, ...){
  if(family$family=="gaussian"){
    # using optimizer bobyqa
    suppressWarnings(
      fit.tweedie <-  cpglm(Y~.,data=X,optimizer = "bobyqa")
    )
    pred <- predict(fit.tweedie, newdata=newX, type="response")
    fit <- list(object = fit.tweedie)
    class(fit) <- "SL.tweedie" 
    out <- list(pred=pred, fit=fit)
    return(out)
  }else{
    stop("SL.tweedie not written for binomial family")
  }
}

predict.SL.tweedie <- function(object, newdata, ...) {
  # newdata must be a dataframe, not a matrix.
  if (is.matrix(newdata)) {
    newdata = as.data.frame(newdata)
  }
  pred <- predict(object = object$object, newdata = newdata, type = "response")
  pred
}

#======================================================================#
# Modified version of SL.caret that prints less annoying GBM output
#======================================================================#
# cv.number = 10

SL.caret1 <- function(Y, X, newX, family, obsWeights, method = "rf", tuneLength = 3, 
                      trControl=trainControl(method = "cv", number = 10, verboseIter = FALSE),
                      metric,...) 
{
  if (length(unique(Y))>2){
    if(is.matrix(Y)) Y <- as.numeric(Y)
    metric <- "RMSE"
    if(method=="gbm"){
      suppressWarnings(
        fit.train <- caret::train(x = X, y = Y, weights = obsWeights, 
                                  metric = metric, method = method, 
                                  tuneLength = tuneLength, 
                                  trControl = trControl,verbose=FALSE)
      )
    }else{
      suppressWarnings(
        fit.train <- caret::train(x = X, y = Y, weights = obsWeights, 
                                  metric = metric, method = method, 
                                  tuneLength = tuneLength, 
                                  trControl = trControl)
      )
    }
    pred <- predict(fit.train, newdata = newX, type = "raw")
  }
  if (length(unique(Y))<=2) {
    metric <- "Accuracy"
    Y.f <- as.factor(Y)
    levels(Y.f) <- c("A0", "A1")
    fit.train <- caret::train(x = X, y = Y.f, weights = obsWeights,
                              metric = metric, method = method, 
                              tuneLength = tuneLength, 
                              trControl = trControl)
    pred <- predict(fit.train, newdata = newX, type = "prob")[,2]
  }
  fit <- list(object = fit.train)
  out <- list(pred = pred, fit = fit)
  class(out$fit) <- c("SL.caret")
  return(out)
}

#==================================================#
# CV-Random Forest
#==================================================#
# tuneLength=3

SL.rf.caret1 <- function(...,method="rf",tuneLength=3){
  SL.caret1(...,method=method,tuneLength=tuneLength)
}


######################################### Two-part model #######################################################

# Part 2:

#=========================================================#
# log-OLS: OLS on ln(y) + smear retransformation
# GLM with Gaussian family and id link on log(Y) + Duan (1983) correction
#=========================================================#

SL.logOLS.smear <- function(Y, X, newX, family, obsWeights, ...){
  if(family$family=="gaussian"){
    logY <- log(Y)
    fit.logGLM <- glm(logY ~ ., data=X, family=family, weights=obsWeights)
    mu <- predict(fit.logGLM, type="response", newdata=X)
    resid <- logY - mu
    pred <- exp(predict(fit.logGLM, type="response",newdata=newX))*mean(exp(resid))
    fit <- list(object=fit.logGLM, mean(exp(resid)))
    class(fit) <- "SL.logOLS.smear"
  }else{
    stop("SL.logGLM.smear not written for binomial family")
  }
  out <- list(fit=fit, pred=pred)
  return(out)
}

# predict function for SL.logOLS.smear
predict.SL.logOLS.smear <- function(object, newdata, ...){
  mu <- predict(object$object, newdata=newdata, type="response")
  correction <- object[[2]]
  return(exp(mu)*correction) 
}

#=========================================================#
# GLM (Gamma distribution + log-link)
#=========================================================#

SL.gammaLogGLM <- function(Y, X, newX, family, obsWeights, ...){
  if(family$family=="gaussian"){
    fit.glm <- glm(Y ~ ., data=X, family=Gamma(link='log'), weights=obsWeights,
                   control=list(maxit=100))
    pred <- predict(fit.glm, newdata=newX, type="response")
    fit <- list(object = fit.glm)
    class(fit) <- "SL.glm" # can use predict.SL.glm
    out <- list(pred=pred, fit=fit)
    return(out)
  }else{
    stop("SL.logGLM not written for binomial family")
  }
}

#=========================================================#
# GLM (Gamma distribution + Identity-link)
#=========================================================#

SL.gammaIdentityGLM <- function(Y, X, newX, family, obsWeights,...){
  if(family$family=="gaussian"){
    fit.glm <- glm(Y ~ ., data=X, family=Gamma(link='identity'), 
                   weights=obsWeights,
                   control=list(maxit=100), start=c(mean(Y),rep(0,ncol(X))))
    pred <- predict(fit.glm, newdata=newX, type="response")
    fit <- list(object = fit.glm)
    class(fit) <- "SL.glm"
    out <- list(pred=pred, fit=fit)
    return(out)
  }else{
    stop("SL.gammaIdentityGLM not written for binomial family")
  }
}

#=========================================================#
# Adaptive GLM algorithm of Manning (2001)
#=========================================================#

SL.manningGLM <- function(Y, X, newX, family, obsWeights, 
                          kCut = 3, # kurtosis cutpoint
                          lambdaCut = c(0.5,1.5,2.5), # skew cutpoint
                          startNLS=0, # starting values for NLS?
                          ...){
  if(family$family=="gaussian"){
    require(moments)
    # first do ols on log scale
    logY <- log(Y)
    fit.logGLM <- glm(logY ~ ., data=X, family=family, weights=obsWeights)
    mu <- predict(fit.logGLM, type="response", newdata=X)
    resid <- logY - mu
    # check kurtosis of residuals
    k <- kurtosis(resid)
    # by default use these methods
    # some of the other GLMs are unstable and if they fail, this 
    # algorithm returns log OLS + smearning estimate
    pred <- exp(predict(fit.logGLM, type="response",newdata=newX))*mean(exp(resid))
    fit <- list(object=fit.logGLM, mean(exp(resid)))
    class(fit) <- "SL.logOLS.smear"
    try({
      if(k < kCut){
        # park test
        fit.initGLM <- glm(Y ~ ., data=X, weights=obsWeights, family="gaussian")
        muPark <- predict(fit.initGLM, type="response", newdata=X)
        resid2Park <- (Y - muPark)^2
        fit.parkGLM <- glm(resid2Park ~ muPark, family="gaussian")
        lambda1 <- fit.parkGLM$coef[2]
        # use nls
        if(lambda1 < lambdaCut[1]){
          xNames <- colnames(X)
          d <- length(xNames)
          bNames <- paste0("b",1:d)
          form <- apply(matrix(1:d), 1, function(i){
            paste(c(bNames[i],xNames[i]),collapse="*")
          })
          formula <- paste(form,collapse=" + ")
          try({
            fit.nls <- nls(as.formula(paste0("Y ~ exp(b0 +",formula,")")), data= data.frame(Y, X),
                           start=eval(parse(text=paste0(
                             "list(b0=0.5,",paste(paste0(bNames, "=", startNLS),collapse=","),")"))))
          })
          pred <- predict(fit.nls, newdata=newX)
          fit <- list(object=fit.nls)
          class(fit) <- "SL.manningGLM"
        }else if(lambda1 < lambdaCut[2] & lambda1 >= lambdaCut[1]){
          # use poisson glm
          fit.poisGLM <- suppressWarnings(
            glm(Y ~ ., data=X, weights=obsWeights, family="poisson",control=list(maxit=100))
          )
          pred <- predict(fit.poisGLM, newdata=newX, type="response")
          fit <- list(object=fit.poisGLM)
          class(fit) <- "SL.manningGLM"
        }else if(lambda1 < lambdaCut[3] & lambda1 >= lambdaCut[2]){
          # use gamma glm
          fit.gammaGLM <- glm(Y ~ ., data=X, weights=obsWeights, family=Gamma(link='log'),control=list(maxit=100))
          pred <- predict(fit.gammaGLM, newdata=newX, type="response")
          fit <- list(object=fit.gammaGLM)
          class(fit) <- "SL.manningGLM"
        }else if(lambda1 > lambdaCut[3]){
          # use inverse gaussian glm -- not very stable
          fit.invGaussianGLM <- glm(Y ~ ., data=X,weights=obsWeights, family=inverse.gaussian(link="log"),control=list(maxit=100))
          pred <- predict(fit.invGaussianGLM, newdata=newX, type="response")
          fit <- list(object=fit.invGaussianGLM)
          class(fit) <- "SL.manningGLM"
        }
      }
    }, silent=TRUE)
  }else{
    stop("SL.manningGLM doesn't work with binomial family.")
  }
  out <- list(pred = pred, fit=fit)
  return(out)
}

# predict function
predict.SL.manningGLM <- function(object, newdata,...){
  if(!is.list(object$object)){
    pred <- predict(object=object$object, newdata=newdata, type="response")
  }else{
    pred <- predict(object=object$object, newdata=newdata, type="response")
  }
  pred
}

#=========================================================#
# Accelerated Failure Time Models (AFT)
#=========================================================#

SL.flexsurvreg <- function(Y, X, newX, family, obsWeights,
                           dist="gengamma",...){
  require(flexsurv)
  if(family$family=="gaussian"){
    fit.flexSurv <- flexsurvreg(
      as.formula(paste0("Surv(Y, rep(1, length(Y))) ~", paste(colnames(X),collapse="+"))) ,
      data=X, dist=dist
    )
    pred <- predict.SL.flexsurvreg(object=list(object=fit.flexSurv), newdata=newX, type="mean")
    fit <- list(object=fit.flexSurv)
    class(fit) <- "SL.flexsurvreg"
    out <- list(fit=fit, pred=pred)
  }else{
    stop("SL.genGamma not implemented for binominal family")
  }
  out
}

#prediction
predict.SL.flexsurvreg <- function(object, newdata, type="mean", ...){
  # function to return survival probability based on flexsurv object
  .getSurv <- function(x, fit, thisnewdata){
    summary(fit, t=x, B=0, newdata=thisnewdata)[[1]][,2]
  }
  pred <- as.numeric(apply(matrix(1:nrow(newdata)), 1, function(i){
    upper <- Inf
    out <- NA; class(out) <- "try-error"
    # integrate can be finnicky, so for stability, we first try to integrate with
    # upper limit = Inf, but if that fails move to 1e8, which sometimes is able to 
    # provide a sane answer when upper limit=Inf fails. Keep trying smaller and smaller
    # values, but don't go smaller than 1e6. If you try, then it just returns a random 
    # number between 0 and 1e6, which prevents Super Learner from crashing. 
    while(class(out)=="try-error" & upper > 1e6){
      out <- try(integrate(.getSurv, fit=object$object, thisnewdata=newdata[i,],lower=0,upper=upper)$value, silent=TRUE)
      if(upper==Inf){
        upper <- 1e8
      }else{
        upper <- upper/2
      }
    }
    if(class(out)=="try-error"){
      warning("Unable to integrate survival function. Returning random number between 0 and 100k")
      out <- runif(1,0,100000)
    }
    out
  }))
  pred
}

#=================================================================#
# Accelerated Failure Time Models (AFT): Genrealized Gamma
#=================================================================#

SL.gengamma <- function(..., dist="gengamma"){
  SL.flexsurvreg(...,dist=dist)
}

#=================================================================#
# Cox Proportional Hazard 
#=================================================================#

SL.coxph  <- function(Y, X, newX, family, obsWeights,
                      dist="gengamma",...){
  if(family$family=="gaussian"){
    library(survival)
    fit.coxph <- coxph(Surv(Y,rep(1,length(Y)))~., data=X)
    fit <- list(object=fit.coxph)
    class(fit) <- "SL.coxph"
    pred <- predict.SL.coxph(object=list(object=fit.coxph), newdata=newX)
  }else{
    stop("SL.coxph not implemented for binominal family")
  }
  return(list(fit=fit,pred=pred))
}

# prediction
predict.SL.coxph <- function(object,newdata,type="mean",...){
  # use surv.fit to get survival estimate and because by default it uses
  # nelson-aalen hazard, easy to convert back to an estimate of the mean
  surv.fit <- survfit(object$object, newdata=newdata)
  pred <- colSums(
    diff(c(0,surv.fit$time))*rbind(
      rep(1,dim(surv.fit$surv)[2]),
      surv.fit$surv[1:(dim(surv.fit$surv)[1]-1),]
    )
  )
  pred
}

#=================================================================#
# Quantile regression method of Wang and Zhou (2009)
#=================================================================#

SL.wangZhou <- function(Y, X, newX, family, obsWeights, 
                        g="log", # transformation of Y
                        m=length(Y), # number of quantiles
                        c=0.2, # for calculating truncated mean
                        b=0.05,# for calculating truncated mean
                        ...){
  require(quantreg)
  if(family$family=="gaussian"){
    n <- length(Y)
    # calculate alpha_n for calculating truncated mean
    alpha <- c*n^(-1/(1+4*b))
    tau <- seq(alpha, 1-alpha, length=m)
    # transform Y
    if(g=="log"){
      thisY <- log(Y)
      ginv <- function(x){ exp(x) }
    }else{
      stop("SL.wangZhou only implemented for log transforms")
    }
    # get quantile regressions
    suppressWarnings(
      fm <- rq(formula=as.formula("thisY~."), tau=tau, weights=obsWeights,
               data=data.frame(thisY,X))
    )
    QhatList <- predict(fm, newdata=newX, stepfun=TRUE, type="Qhat")
    QhatRearrange <- lapply(QhatList, rearrange)
    # transform to means
    pred <- unlist(lapply(QhatRearrange, FUN=function(Q){
      Qw <- ginv(environment(Q)$y[-which(duplicated(environment(Q)$x))])
      1/(1-2*alpha) * sum(Qw * diff(c(0,tau)))
    }))
  }else{
    stop("SL.wangZhou not written for binomial family")
  }
  fit <- list(object=fm, alpha=alpha, ginv=ginv)
  class(fit) <- "SL.wangZhou"
  out <- list(pred=pred, fit=fit)
  return(out)
}

# predict function for SL.wangZhou
predict.SL.wangZhou <- function(object, newdata, ...){
  require(quantreg)
  QhatList <- predict(object$object, newdata=newdata, stepfun=TRUE, type="Qhat")
  QhatRearrange <- lapply(QhatList, rearrange)
  pred <- mapply(Q=QhatRearrange, dt=diff(c(0,object$object$tau)), function(Q,dt){
    Qw <- do.call(object$ginv,args=list(x=environment(Q)$y[-which(duplicated(environment(Q)$x))]))
    1/(1-2*object$alpha) * sum(Qw * dt)
  })    
  pred
}

#==========================================================================================#
# Discrete conditional density estimator /  Adaptive Hazard method (Gilleskie & Mroz)
#==========================================================================================#

SL.gilleskie <- function(Y, X, newX, family, obsWeights,
                         kValues=c(5,15,25), # number of intervals
                         yBoundaries, # boundaries on the y-variable
                         maxPoly=2, # maximum polynomial in hazard regressions
                         ...){
  # need the sandwich package for covariate selection algorithm
  library(sandwich)
  # maxPoly describes the polynomial used for the partition variable
  # in the hazard regression. Choosing the number of partitions to be
  # less than this value leads to rank definiciency in glm()
  if(any(kValues < maxPoly)){ 
    warning("kValue specified that is less than maxPoly. These kValues will be ignored")
    kValues <- kValues[kValues>maxPoly]
  }
  
  #====================================================
  # get hazard fit over different partitions of data
  #====================================================
  
  outList <- lapply(split(kValues, 1:length(kValues)), FUN=function(K){
    # break up Y into K+1 partitions
    Ytilde <- cut(Y, breaks=quantile(Y, p=seq(0,1,length=K+1)), labels=FALSE,
                  include.lowest=TRUE)
    # make a long versions data set
    longDat <- data.frame(Ytilde,X,id=1:length(Y))[rep(1:length(Y),Ytilde),]
    # assign parition number variable
    row.names(longDat)[row.names(longDat) %in% paste(row.names(data.frame(Ytilde,X)))] <- paste(row.names(data.frame(Ytilde,X)),".0",sep="")  
    longDat$k <- as.numeric(paste(unlist(strsplit(row.names(longDat),".",fixed=T))[seq(2,nrow(longDat)*2,2)]))+1
    # indicator of falling in a particular partition
    longDat$indY <- as.numeric(longDat$k==longDat$Ytilde)
    # loop to do covariate selection
    pVal <- Inf
    d <- maxPoly
    while(pVal > 0.05 & d>=1){
      # generate the regression equation
      rhs <- NULL
      for(i in 1:(ncol(X)-1)){
        rhs <- c(rhs, paste0("poly(",colnames(X)[i],",",ifelse(length(unique(X[,i]))>d, d, length(unique(X[,i]))-1),")*poly(k,",d,")*",colnames(X)[(i+1):(ncol(X))],collapse="+"))
      }
      rhs <- c(rhs, paste0("poly(",colnames(X)[ncol(X)],",",ifelse(length(unique(X[,i]))>d, d, length(unique(X[,i]))-1),")*poly(k,",d,")"))
      # fit the hazard regression
      suppressWarnings(
        fm <- glm(as.formula(paste0("indY ~ ",paste0(rhs,collapse="+"))),
                  data=longDat, family="binomial")
      )
      # get coefficients of degree d
      dropNum <- NULL
      for(cn in colnames(X)){
        dropNum <- c(dropNum, grep(paste0(cn,", ",d,")",d), names(fm$coef[!is.na(fm$coef)])))
      }
      dropCoef <- fm$coef[!is.na(fm$coef)][dropNum]
      # get covariance matrix for all thos ecoefficients
      fullCov <- vcovHC(fm,type="HC0")
      dropCov <- fullCov[dropNum,dropNum]
      # test significance of those coefficients
      chi2Stat <- tryCatch({
        t(dropCoef)%*%solve(dropCov)%*%dropCoef
      },error=function(e){ return(0) }
      )
      pVal <- pchisq(chi2Stat, lower.tail=FALSE, df=length(dropCoef))
      d <- d-1
    }
    # after done dropping polynomial terms, get hazard predictions
    suppressWarnings(
      longDat$haz <- predict(fm, newdata=longDat,type="response")
    )
    # calculate likelihood
    tmp <- by(longDat, factor(longDat$id), FUN=function(x){
      prod((c(1,cumprod(1-x$haz[x$k < x$Ytilde])) * x$haz)^x$indY)
    })
    LKR <- sum(log(as.numeric(tmp))) + length(Y)*log(K)
    # return the likelihood ratio and estimate hazard regression
    return(list(LKR=LKR, fm=fm))
  })
  # figure out which one had highest likelihood ratio
  LKRs <- unlist(lapply(outList, function(x){x[[1]]}))
  maxLKR <- which(LKRs==max(LKRs))
  maxK <- kValues[maxLKR]
  thisOut <- outList[[maxLKR]]
  # get mean in each partition for transforming back to mean-scale
  Ytilde <- cut(Y, breaks=quantile(Y, p=seq(0,1,length=maxK+1)), labels=FALSE,
                include.lowest=TRUE)
  hK <- apply(matrix(1:maxK), 1, function(x){
    mean(Y[Ytilde==x])
  })
  # calculate mean by calculating density of each partition 
  pred <- apply(matrix(1:length(newX[,1])), 1, FUN=function(x){
    suppressWarnings(
      haz <- predict(thisOut$fm, 
                     newdata=data.frame(newX[x,],k=1:maxK),
                     type="response")
    )
    dens <- c(1,cumprod(1-haz[1:(maxK-1)])) * haz
    sum(hK*dens)
  })
  fit <- list(object=thisOut$fm, maxK=maxK, hK=hK)
  class(fit) <- "SL.gilleskie"
  out <- list(pred=pred, fit=fit)
  out
}

# predict function for SL.gilleskie
predict.SL.gilleskie <- function(object, newdata,...){
  pred <- apply(matrix(1:length(newdata[,1])), 1, FUN=function(x){
    suppressWarnings(
      haz <- predict(object$object,newdata=data.frame(newdata[rep(x,object$maxK),],k=1:object$maxK),
                     type="response")
    )
    dens <- c(1,cumprod(1-haz[1:(object$maxK-1)])) * haz
    sum(object$hK*dens)
  })
  pred
}






# Initial code for two-stage super learner (From twostageSL.R)



############################ Scaled quadratic programming ##################################
## function for generating weights (coefficients): scaled quadratic programming

method.CC_LS.scale <- function() {
  computeCoef = function(Z, Y, libraryNames, verbose,
                         obsWeights=rep(1, length(Y)),
                         errorsInLibrary = NULL, ...) {
    # compute cvRisk
    cvRisk <- apply(Z, 2, function(x) mean(obsWeights*(x-Y)^2))
    names(cvRisk) <- libraryNames
    # compute coef
    compute <- function(x, y, wt=rep(1, length(y))) {
      wX <- sqrt(wt) * x
      wY <- sqrt(wt) * y
      D <- crossprod(wX)
      d <- crossprod(wX, wY)
      A <- cbind(rep(1, ncol(wX)), diag(ncol(wX)))
      bvec <- c(1, rep(0, ncol(wX)))
      sc <- norm(D,"2")
      # scale D matrix & d vector to aviod inconsistent constraints
      fit <- quadprog::solve.QP(Dmat=D/sc, dvec=d/sc, Amat=A, bvec=bvec, meq=1,
                                factorized = F)
      invisible(fit)
    }
    modZ <- Z
    # check for columns of all zeros. assume these correspond
    # to errors that SuperLearner sets equal to 0. not a robust
    # solution, since in theory an algorithm could predict 0 for
    # all observations (e.g., SL.mean when all Y in training = 0)
    naCols <- which(apply(Z, 2, function(z){ all(z == 0 ) }))
    anyNACols <- length(naCols) > 0
    if(anyNACols){
      # if present, throw warning identifying learners
      warning(paste0(paste0(libraryNames[naCols],collapse = ", "), " have NAs.",
                     "Removing from super learner."))
    }
    # check for duplicated columns
    # set a tolerance level to avoid numerical instability
    tol <- 8
    dupCols <- which(duplicated(round(Z, tol), MARGIN = 2))
    anyDupCols <- length(dupCols) > 0
    if(anyDupCols){
      # if present, throw warning identifying learners
      warning(paste0(paste0(libraryNames[dupCols],collapse = ", "),
                     " are duplicates of previous learners.",
                     " Removing from super learner."))
    }
    # remove from Z if present
    if(anyDupCols | anyNACols){
      rmCols <- unique(c(naCols,dupCols))
      modZ <- Z[,-rmCols]
    }
    # compute coefficients on remaining columns
    fit <- compute(x = modZ, y = Y, wt = obsWeights)
    coef <- fit$solution
    if (anyNA(coef)) {
      warning("Some algorithms have weights of NA, setting to 0.")
      coef[is.na(coef)] = 0
    }
    # add in coefficients with 0 weights for algorithms with NAs
    if(anyDupCols | anyNACols){
      ind <- c(seq_along(coef), rmCols - 0.5)
      coef <- c(coef, rep(0, length(rmCols)))
      coef <- coef[order(ind)]
    }
    # Set very small coefficients to 0 and renormalize.
    coef[coef < 1.0e-4] <- 0
    coef <- coef / sum(coef)
    if(!sum(coef) > 0) warning("All algorithms have zero weight", call. = FALSE)
    list(cvRisk = cvRisk, coef = coef, optimizer = fit)
  }
  
  computePred = function(predY, coef, ...) {
    predY %*% matrix(coef)
  }
  out <- list(require = "quadprog",
              computeCoef = computeCoef,
              computePred = computePred)
  invisible(out)
}

## Two-Stage SuperLearner
##############################################################################################

twostageSL <- function(Y, X, newX = NULL, library.2stage, library.1stage,twostage,
                       family.1, family.2, family.single, method="method.CC_LS",
                       id=NULL, verbose=FALSE, control = list(), cvControl = list(),
                       obsWeights = NULL, env = parent.frame()){
  
  # Begin timing how long two-stage SuperLearner takes to execute
  time_start = proc.time()
  
  # Get details of estimation algorithm for the algorithm weights (coefficients)
  if (is.character(method)) {
    if (exists(method, mode = 'list')) {
      method <- get(method, mode = 'list')
    } else if (exists(method, mode = 'function')) {
      method <- get(method, mode = 'function')()
    }
  } else if (is.function(method)) {
    method <- method()
  }
  # make some modifications (scale) to the superlearner:method.CC_LS
  method$computeCoef <- method.CC_LS.scale()$computeCoef
  if(!is.list(method)) {
    stop("method is not in the appropriate format. Check out help('method.template')")
  }
  if(!is.null(method$require)) {
    sapply(method$require, function(x) require(force(x), character.only = TRUE))
  }
  
  # get defaults for controls and make sure in correct format
  control <- do.call('SuperLearner.control', control)
  # change the logical for saveCVFitLibrary to TRUE (we are gonna use that)
  control$saveCVFitLibrary <- TRUE
  
  cvControl <- do.call('SuperLearner.CV.control', cvControl)
  
  # put together the library
  library.stage1 <- library.2stage$stage1
  library.stage2 <- library.2stage$stage2
  library.stage_1 <- SuperLearner:::.createLibrary(library.stage1)
  library.stage_2 <- SuperLearner:::.createLibrary(library.stage2)
  library.stage_single <- SuperLearner:::.createLibrary(library.1stage)
  SuperLearner:::.check.SL.library(library = c(unique(library.stage_1$library$predAlgorithm),
                                               library.stage_1$screenAlgorithm))
  SuperLearner:::.check.SL.library(library = c(unique(library.stage_2$library$predAlgorithm),
                                               library.stage_2$screenAlgorithm))
  SuperLearner:::.check.SL.library(library = c(unique(library.stage_single$library$predAlgorithm),
                                               library.stage_single$screenAlgorithm))
  call <- match.call(expand.dots = TRUE)
  
  # should we be checking X and newX for data.frame?
  # data.frame not required, but most of the built-in wrappers assume a data.frame
  if(!inherits(X, 'data.frame')) message('X is not a data frame. Check the algorithms in SL.library to make sure they are compatible with non data.frame inputs')
  varNames <- colnames(X)
  N <- dim(X)[1L]
  p <- dim(X)[2L]
  k.1 <- nrow(library.stage_1$library)
  k.2 <- nrow(library.stage_2$library)
  k.single <- nrow(library.stage_single$library)
  k.2stage <- k.1*k.2
  k.all <- k.1*k.2+k.single
  kScreen.1 <- length(library.stage_1$screenAlgorithm)
  kScreen.2 <- length(library.stage_2$screenAlgorithm)
  kScreen.single <- length(library.stage_single$screenAlgorithm)
  kscreen.2stage <- kScreen.1*kScreen.2
  kscreen.all <- kScreen.1*kScreen.2+kScreen.single
  
  # family can be either character or function, so these lines put everything together
  #family for stage 1
  if(is.character(family.1))
    family.1 <- get(family.1, mode="function", envir=parent.frame())
  if(is.function(family.1))
    family.1 <- family.1()
  if (is.null(family.1$family)) {
    print(family.1)
    stop("'family' not recognized")
  }
  # family for stage 2
  if(is.character(family.2))
    family.2 <- get(family.2, mode="function", envir=parent.frame())
  if(is.function(family.2))
    family.2 <- family.2()
  if (is.null(family.2$family)) {
    print(family.2)
    stop("'family' not recognized")
  }
  # family for single stage
  if(is.character(family.single))
    family.single <- get(family.single, mode="function", envir=parent.frame())
  if(is.function(family.single))
    family.single <- family.single()
  if (is.null(family.single$family)) {
    print(family.single)
    stop("'family' not recognized")
  }
  
  # check if the model use method.AUC
  if (family.1$family != "binomial" & isTRUE("cvAUC" %in% method$require)){
    stop("'method.AUC' is designed for the 'binomial' family only")
  }
  if (family.2$family != "binomial" & isTRUE("cvAUC" %in% method$require)){
    stop("'method.AUC' is designed for the 'binomial' family only")
  }
  if (family.single$family != "binomial" & isTRUE("cvAUC" %in% method$require)){
    stop("'method.AUC' is designed for the 'binomial' family only")
  }
  
  # chekc whether screen algorithm compatible with number of columns
  if(p < 2 & !identical(library.stage_1$screenAlgorithm, "All")) {
    warning('Screening algorithms specified in combination with single-column X.')
  }
  if(p < 2 & !identical(library.stage_2$screenAlgorithm, "All")) {
    warning('Screening algorithms specified in combination with single-column X.')
  }
  if(p < 2 & !identical(library.stage_single$screenAlgorithm, "All")) {
    warning('Screening algorithms specified in combination with single-column X.')
  }
  
  # generate library names
  # stage 1
  libname.stage.1 <- NULL
  lib.stage1 <- library.stage_1$library$predAlgorithm
  lib.stage1.screen <- library.stage_1$screenAlgorithm[library.stage_1$library$rowScreen]
  repname <- function(x) {
    name <- rep(x,k.2)
    return(name)
  }
  libname.stage.1 <- unlist(lapply(lib.stage1,repname), use.names=FALSE)
  libname.stage.1.screen <- unlist(lapply(lib.stage1.screen,repname), use.names=FALSE)
  # stage 2
  libname.stage.2 <- NULL
  lib.stage2 <- library.stage_2$library$predAlgorithm
  lib.stage2.screen <- library.stage_2$screenAlgorithm[library.stage_2$library$rowScreen]
  libname.stage.2 <- rep(lib.stage2,k.1)
  libname.stage.2.screen <- rep(lib.stage2.screen,k.1)
  # single stage
  libname.stage.single <- library.stage_single$library$predAlgorithm
  libname.stage.single.screen <- library.stage_single$screenAlgorithm[library.stage_single$library$rowScreen]
  
  twostage.library <- paste("S1:",paste(libname.stage.1,libname.stage.1.screen,sep="_"),
                            "+ S2:",paste(libname.stage.2,libname.stage.2.screen,sep="_"))
  wholelibrary <- c(twostage.library,
                    paste("Single:",paste(libname.stage.single,libname.stage.single.screen,sep="_")))
  
  # add family for two stages and single stage
  family <- list(stage1 = family.1,stage2 = family.2,
                 stage.single = family.single)
  
  # add library for two stages
  lib <- list(twostage=data.frame("predAlgorithm"=paste("S1:",libname.stage.1,
                                                        "+ S2:",libname.stage.2),
                                  "rowScreen.Stage.1"=rep(library.stage_1$library$rowScreen,each=k.2),
                                  "rowScreen.Stage.2"=rep(library.stage_2$library$rowScreen,k.1)),
              singlestage=library.stage_single$library)
  library <- list("library"=lib,
                  "screenAlgorithm"=list(stage.1 = library.stage_1$screenAlgorithm,
                                         stage.2 = library.stage_2$screenAlgorithm,
                                         stage.single = library.stage_single$screenAlgorithm))
  
  # if newX is missing, use X
  if(is.null(newX)) {
    newX <- X
  }
  
  # Various chekcs for data structure
  if(!identical(colnames(X), colnames(newX))) {
    stop("The variable names and order in newX must be identical to the variable names and order in X")
  }
  if (sum(is.na(X)) > 0 | sum(is.na(newX)) > 0 | sum(is.na(Y)) > 0) {
    stop("missing data is currently not supported. Check Y, X, and newX for missing values")
  }
  if (!is.numeric(Y)) {
    stop("the outcome Y must be a numeric vector")
  }
  
  # errors records if an algorithm stops either in the CV step and/or in full data
  errorsInCVLibrary <- rep(0, k.all)
  errorsInLibrary <- rep(0, k.all)
  
  ########################################################################################################
  # Step 0: make valid rows
  # ensure each folds have approximately equal number of obs with y=0
  V <- cvControl$V
  ord <- order(Y)
  cvfold <- rep(c(1:V,V:1),N)[1:N]
  folds <- split(ord, factor(cvfold))
  folds <- lapply(folds,sort,decreasing=FALSE)
  # check
  tab <- rep(NA,V)
  for (i in 1:V) {
    tab[i] <- sum(Y[folds[[i]]]==0)
  }
  num.0 <- data.frame("fold"=paste0("fold ",c(1:cvControl$V)),"number.of.0"=tab)
  
  cvControl$validRows = folds
  
  # test id
  if(is.null(id)) {
    id <- seq(N)
  }
  if(!identical(length(id), N)) {
    stop("id vector must have the same dimension as Y")
  }
  # test observation weights
  if(is.null(obsWeights)) {
    obsWeights <- rep(1, N)
  }
  if(!identical(length(obsWeights), N)) {
    stop("obsWeights vector must have the same dimension as Y")
  }
  
  #########################################################################################################
  # Step 1: fit superlearner for modeling prob of y=0
  time_train_start = proc.time()
  
  # list all the algorithms considered
  # save cross-validated fits (10) in the control option
  step1.fit <- SuperLearner(Y=as.numeric(Y==0),X=X,family=family.1,
                            SL.library=library.stage1,verbose=verbose,
                            method=method.CC_nloglik,
                            control=list(saveCVFitLibrary=T),
                            cvControl=cvControl)
  # get the cross-validated predicted values for each algorithm in SL.library
  # P(Y=0|X)
  z1 <- step1.fit$Z
  # get the cross-validated fits (10 fits for 10 training set) for each algorithm
  stage1.cvFitLibrary <- step1.fit$cvFitLibrary
  
  
  ##########################################################################################################
  # step 2: fit model for E[Y|Y>0,X]
  # create function for the cross-validation step at stage 2:
  .crossValFUN <- function(valid, Y, dataX, predX, id, obsWeights, library, family,
                           kScreen, k, p, libraryNames, saveCVFitLibrary) {
    tempLearn <- dataX[-valid, , drop = FALSE]
    tempOutcome <- Y[-valid]
    tempValid <- predX[valid, , drop = FALSE]
    tempWhichScreen <- matrix(NA, nrow = kScreen, ncol = p)
    tempId <- id[-valid]
    tempObsWeights <- obsWeights[-valid]
    
    # create subset with only obs y>0
    pid <- row.names(tempLearn)
    dat.p <- cbind(pid,tempLearn,tempOutcome)
    tempOutcome.p <- tempOutcome[tempOutcome>0]
    tempLearn.p <- dat.p[dat.p$tempOutcome>0,-c(1,ncol(dat.p))]
    tempId.p <- dat.p[dat.p$tempOutcome>0,1]
    tempObsWeights.p <- obsWeights[tempId.p]
    
    # should this be converted to a lapply also?
    for(s in seq(kScreen)) {
      screen_fn = get(library$screenAlgorithm[s], envir = env)
      testScreen <- try(do.call(screen_fn,
                                list(Y = tempOutcome.p,
                                     X = tempLearn.p,
                                     family = family,
                                     id = tempId.p,
                                     obsWeights = tempObsWeights.p)))
      if(inherits(testScreen, "try-error")) {
        warning(paste("replacing failed screening algorithm,", library$screenAlgorithm[s], ", with All()", "\n "))
        tempWhichScreen[s, ] <- TRUE
      } else {
        tempWhichScreen[s, ] <- testScreen
      }
      if(verbose) {
        message(paste("Number of covariates in ", library$screenAlgorithm[s], " is: ", sum(tempWhichScreen[s, ]), sep = ""))
      }
    } #end screen
    
    # should this be converted to a lapply also?
    out <- matrix(NA, nrow = nrow(tempValid), ncol = k)
    if(saveCVFitLibrary){
      model_out <- vector(mode = "list", length = k)
    }else{
      model_out <- NULL
    }
    
    for(s in seq(k)) {
      pred_fn = get(library$library$predAlgorithm[s], envir = env)
      testAlg <- try(do.call(pred_fn,
                             list(Y = tempOutcome.p,
                                  X = subset(tempLearn.p,
                                             select = tempWhichScreen[library$library$rowScreen[s], ],
                                             drop=FALSE),
                                  newX = subset(tempValid,
                                                select = tempWhichScreen[library$library$rowScreen[s], ],
                                                drop=FALSE),
                                  family = family,
                                  id = tempId.p,
                                  obsWeights = tempObsWeights.p)))
      if(inherits(testAlg, "try-error")) {
        warning(paste("Error in algorithm", library$library$predAlgorithm[s], "\n  The Algorithm will be removed from the Super Learner (i.e. given weight 0) \n" ))
        # errorsInCVLibrary[s] <<- 1
      } else {
        out[, s] <- testAlg$pred
        if(saveCVFitLibrary){
          model_out[[s]] <- testAlg$fit
        }
      }
      if (verbose) message(paste("CV", libraryNames[s]))
    } #end library
    if(saveCVFitLibrary){
      names(model_out) <- libraryNames
    }
    invisible(list(out = out, model_out = model_out))
  }
  
  # the lapply performs the cross-validation steps to create Z for stage 2
  # additional steps to put things in the correct order
  # rbind unlists the output from lapply
  # need to unlist folds to put the rows back in the correct order
  
  crossValFUN_out <- lapply(folds, FUN = .crossValFUN,
                            Y = Y, dataX = X, predX = X, id = id,
                            obsWeights = obsWeights, family = family.2,
                            library = library.stage_2, kScreen = kScreen.2,
                            k = k.2, p = p, libraryNames = library.stage_2$library$predAlgorithm,
                            saveCVFitLibrary = control$saveCVFitLibrary)
  
  # create matrix to store results
  z2 <- matrix(NA,nrow = N,ncol=k.2)
  z2[unlist(folds, use.names = FALSE), ] <- do.call('rbind', lapply(crossValFUN_out, "[[", "out"))
  
  if(control$saveCVFitLibrary){
    stage2.cvFitLibrary <- lapply(crossValFUN_out, "[[", "model_out")
  }else{
    stage2.cvFitLibrary <- NULL
  }
  
  # z1 for E[P(Y=0|X)]
  # z2 for E[Y|Y>0,X]
  # multiply (1-z1)*z2 to generate z
  z <- NULL
  for (i in 1:k.1){
    for (j in 1:k.2){
      temp <- rep(0,N)
      for (k in 1:N){
        temp[k] <- (1-z1[k,i])*z2[k,j]
      }
      z <- cbind(z,temp)
    }
  }
  
  ########################################################################################################
  # step 3: fit the whole model using one stage option (rather than two stages)
  # list all the algorithms considered
  # save cross-validated fits (10) in the control option
  onestage.fit <- SuperLearner(Y=Y,X=X,family=family.single,
                               SL.library=library.1stage,verbose=verbose,
                               method=method.CC_LS.scale,
                               control=list(saveCVFitLibrary=T),
                               cvControl=cvControl)
  # get the cross-validated predicted values for each algorithm in SL.library
  z.single <- onestage.fit$Z
  # get the cross-validated fits (10 fits for 10 training set) for each algorithm
  single.stage.cvFitLibrary <- onestage.fit$cvFitLibrary
  
  # combine 2 stages output z with 1 stage prediction output z
  z <- cbind(z,z.single)
  
  # Check for errors. If any algorithms had errors, replace entire column with
  # 0 even if error is only in one fold.
  errorsInCVLibrary <- apply(z, 2, function(x) anyNA(x))
  if (sum(errorsInCVLibrary) > 0) {
    z[, as.logical(errorsInCVLibrary)] <- 0
  }
  if (all(z == 0)) {
    stop("All algorithms dropped from library")
  }
  
  ########################################################################################################
  # step 4: use cross-validation to calcualte weights for different algorithm at stage 1 & 2
  # using an scaled method.CC_LS in superlearner
  # get optimum weights for each algorithm
  getCoef <- method.CC_LS.scale()$computeCoef(Z=z,Y=Y,libraryNames=wholelibrary,
                                              verbose=verbose)
  coef <- getCoef$coef
  names(coef) <- wholelibrary
  
  time_train = proc.time() - time_train_start
  
  # Set a default in case the method does not return the optimizer result.
  if (!("optimizer" %in% names(getCoef))) {
    getCoef["optimizer"] <- NA
  }
  
  #########################################################################################################
  # step 5: now fit all algorithms in library on entire data set (X) and predict on newX
  
  .screenFun <- function(fun, list) {
    screen_fn = get(fun, envir = env)
    testScreen <- try(do.call(screen_fn, list))
    if (inherits(testScreen, "try-error")) {
      warning(paste("replacing failed screening algorithm,", fun, ", with All() in full data", "\n "))
      out <- rep(TRUE, ncol(list$X))
    } else {
      out <- testScreen
    }
    return(out)
  }
  
  time_predict_start = proc.time()
  
  # stage 1
  whichScreen.stage1 <- sapply(library$screenAlgorithm$stage.1, FUN = .screenFun,
                               list = list(Y = Y, X = X, family = family, id = id, obsWeights = NULL),
                               simplify = FALSE)
  whichScreen.stage1 <- do.call(rbind, whichScreen.stage1)
  # stage 2
  whichScreen.stage2 <- sapply(library$screenAlgorithm$stage.2, FUN = .screenFun,
                               list = list(Y = Y, X = X, family = family, id = id, obsWeights = NULL),
                               simplify = FALSE)
  whichScreen.stage2 <- do.call(rbind, whichScreen.stage2)
  # single stage
  whichScreen.stage.single <- sapply(library$screenAlgorithm$stage.single, FUN = .screenFun,
                                     list = list(Y = Y, X = X, family = family, id = id, obsWeights = NULL),
                                     simplify = FALSE)
  whichScreen.stage.single <- do.call(rbind, whichScreen.stage.single)
  # combine together
  whichScreen <- list(stage1 = whichScreen.stage1,
                      stage2 = whichScreen.stage2,
                      single.stage = whichScreen.stage.single)
  
  # Prediction for each algorithm
  .predFun <- function(index, lib, Y, dataX, newX, whichScreen, family, id, obsWeights,
                       verbose, control, libraryNames) {
    pred_fn = get(lib$predAlgorithm[index], envir = env)
    testAlg <- try(do.call(pred_fn, list(Y = Y,
                                         X = subset(dataX,
                                                    select = whichScreen[lib$rowScreen[index], ], drop=FALSE),
                                         newX = subset(newX, select = whichScreen[lib$rowScreen[index], ], drop=FALSE),
                                         family = family, id = id, obsWeights = obsWeights)))
    # testAlg <- try(do.call(lib$predAlgorithm[index], list(Y = Y, X = dataX[, whichScreen[lib$rowScreen[index], drop = FALSE]], newX = newX[, whichScreen[lib$rowScreen[index], drop = FALSE]], family = family, id = id, obsWeights = obsWeights)))
    if (inherits(testAlg, "try-error")) {
      warning(paste("Error in algorithm", lib$predAlgorithm[index], " on full data", "\n  The Algorithm will be removed from the Super Learner (i.e. given weight 0) \n" ))
      out <- rep.int(NA, times = nrow(newX))
    } else {
      out <- testAlg$pred
      if (control$saveFitLibrary) {
        eval(bquote(fitLibrary[[.(index)]] <- .(testAlg$fit)), envir = fitLibEnv)
      }
    }
    if (verbose) {
      message(paste("full", libraryNames[index]))
    }
    invisible(out)
  }
  
  # stage 1
  # put fitLibrary at stage 1 in it's own environment to locate later
  fitLibEnv <- new.env()
  assign('fitLibrary', vector('list', length = k.1), envir = fitLibEnv)
  assign('libraryNames', library.stage_1$library$predAlgorithm, envir = fitLibEnv)
  evalq(names(fitLibrary) <- library.stage_1$library$predAlgorithm, envir = fitLibEnv)
  # get prediction for stage 1
  predY.stage1 <- do.call('cbind', lapply(seq(k.1), FUN = .predFun,
                                          lib = library.stage_1$library, Y = as.numeric(Y==0), dataX = X,
                                          newX = newX, whichScreen = whichScreen$stage1,
                                          family = family.1, id = id,
                                          obsWeights = obsWeights, verbose = verbose,
                                          control = control,
                                          libraryNames = library.stage_1$library$predAlgorithm))
  # save fit library for stage 1
  stage1.fitlib <- get("fitLibrary",envir = fitLibEnv)
  
  # stage 2
  # put fitLibrary at stage 2 in it's own environment to locate later
  fitLibEnv <- new.env()
  assign('fitLibrary', vector('list', length = k.2), envir = fitLibEnv)
  assign('libraryNames', library.stage_2$library$predAlgorithm, envir = fitLibEnv)
  evalq(names(fitLibrary) <- library.stage_2$library$predAlgorithm, envir = fitLibEnv)
  # get prediction for stage 2
  # create subset with only obs y>0
  pid <- c(1:N)
  dat.p <- cbind(pid,X,Y)
  Y.p <- Y[Y>0]
  X.p <- dat.p[dat.p$Y>0,-c(1,ncol(dat.p))]
  p.id <- dat.p[dat.p$Y>0,1]
  p.obsWeights <- obsWeights[p.id]
  
  predY.stage2 <- do.call('cbind', lapply(seq(k.2), FUN = .predFun,
                                          lib = library.stage_2$library, Y = Y.p, dataX = X.p,
                                          newX = newX, whichScreen = whichScreen$stage2,
                                          family = family.2, id = p.id,
                                          obsWeights = p.obsWeights, verbose = verbose,
                                          control = control,
                                          libraryNames = library.stage_2$library$predAlgorithm))
  # save fit library for stage 2
  stage2.fitlib <- get("fitLibrary",envir = fitLibEnv)
  
  # single stage
  # put fitLibrary at single in it's own environment to locate later
  fitLibEnv <- new.env()
  assign('fitLibrary', vector('list', length = k.single), envir = fitLibEnv)
  assign('libraryNames', library.stage_single$library$predAlgorithm, envir = fitLibEnv)
  evalq(names(fitLibrary) <- library.stage_single$library$predAlgorithm, envir = fitLibEnv)
  # save fit library for single stage
  predY.stage.single <- do.call('cbind', lapply(seq(k.single), FUN = .predFun,
                                                lib = library.stage_single$library, Y = Y, dataX = X,
                                                newX = newX, whichScreen = whichScreen$single.stage,
                                                family = family.single, id = id,
                                                obsWeights = obsWeights, verbose = verbose,
                                                control = control,
                                                libraryNames = library.stage_single$library$predAlgorithm))
  # save fit library for single stage
  stage.single.fitlib <- get("fitLibrary",envir = fitLibEnv)
  
  # get prediction for 2-stage model
  predY <- NULL
  for (i in 1:k.1){
    for (j in 1:k.2){
      pred <- (1-predY.stage1[,i])*predY.stage2[,j]
      predY <- cbind(predY,pred)
    }
  }
  # combine with prediction from singe-stage model
  predY <- cbind(predY,predY.stage.single)
  
  #generate Fitlibrary
  fitLibrary = list("stage1"=stage1.fitlib,
                    "stage2"=stage2.fitlib,
                    "stage.sinlge"=stage.single.fitlib)
  
  #generate cross-validation Fitlibrary
  cvfitLibrary <- list("stage1"=stage1.cvFitLibrary,
                       "stage2"=stage2.cvFitLibrary,
                       "stage.single"=single.stage.cvFitLibrary)
  
  # check for errors
  errorsInLibrary <- apply(predY, 2, function(algorithm) anyNA(algorithm))
  if (sum(errorsInLibrary) > 0) {
    if (sum(coef[as.logical(errorsInLibrary)]) > 0) {
      warning(paste0("Re-running estimation of coefficients removing failed algorithm(s)\n",
                     "Original coefficients are: \n", paste(coef, collapse = ", "), "\n"))
      z[, as.logical(errorsInLibrary)] <- 0
      if (all(z == 0)) {
        stop("All algorithms dropped from library")
      }
      getCoef <- method$computeCoef(Z = z, Y = Y, libraryNames = wholelibrary,
                                    obsWeights = obsWeights, control = control,
                                    verbose = verbose,
                                    errorsInLibrary = errorsInLibrary)
      coef <- getCoef$coef
      names(coef) <- wholelibrary
    } else {
      warning("Coefficients already 0 for all failed algorithm(s)")
    }
  }
  
  # Compute super learner predictions on newX.
  getPred <- method$computePred(predY = predY, coef = coef, control=control)
  
  time_predict = proc.time() - time_predict_start
  
  # Add names of algorithms to the predictions.
  colnames(predY) <- wholelibrary
  
  # Clean up when errors in library.
  if(sum(errorsInCVLibrary) > 0) {
    getCoef$cvRisk[as.logical(errorsInCVLibrary)] <- NA
  }
  
  # Finish timing the full SuperLearner execution.
  time_end = proc.time()
  
  # Compile execution times.
  times = list(everything = time_end - time_start,
               train = time_train,
               predict = time_predict)
  
  # number of algorithms used in each stage
  library.num <- list(stage1 = k.1,
                      stage2 = k.2,
                      stage.single = k.single)
  
  #original library for each stage's algorithm
  orig.library <- list(stage1 = library.stage_1,
                       stage2 = library.stage_2,
                       stage.single = library.stage_single)
  
  # Output whether two-stage resutls or one-stage results
  if (twostage){
    # results
    data.frame("CV.Risk"=getCoef$cvRisk,"Coef"=coef)
    
    # Put everything together in a list.
    out <- list(
      call = call,
      libraryNames = wholelibrary,
      library.Num = library.num,
      orig.library = orig.library,
      SL.library = library,
      SL.predict = getPred,
      coef = coef,
      library.predict = predY,
      Z = z,
      cvRisk = getCoef$cvRisk,
      family = family,
      fitLibrary = fitLibrary,
      cvfitLibrary = cvfitLibrary,
      varNames = varNames,
      validRows = folds,
      number0 = num.0,
      method = method,
      whichScreen = whichScreen,
      control = control,
      cvControl = cvControl,
      errorsInCVLibrary = errorsInCVLibrary,
      errorsInLibrary = errorsInLibrary,
      metaOptimizer = getCoef$optimizer,
      env = env,
      times = times
    )
    class(out) <- c("SuperLearner")
    out
  } else {
    # results
    onestage.fit
    # Put everything together in a list.
  }
}



# Loading MEPS train_data (from MEPS.R) using project GitHub file paths

# train
train <- read.csv("https://github.com/jficorilli/DS_340W_Project/raw/refs/heads/main/Replicating%20Parent%20Paper/train.csv")
# test
test <- read.csv("https://github.com/jficorilli/DS_340W_Project/raw/refs/heads/main/Replicating%20Parent%20Paper/test.csv") %>% select(!c("X"))


######Establishing parallel processing (personal addition):


# Detect total logical cores
total_logical_cores <- detectCores()

# Leave 2 physical cores
n_cores_to_use <- total_logical_cores - 4

# Establish and register cluster
cl <- makeCluster(n_cores_to_use)
registerDoParallel(cl)


# Run computationally intensive code in parallel
tryCatch({
  #==================================================================================#
  # Fit two stage Super Learner
  #==================================================================================#
  
  # fit two-stage superlearner 
  twostage.fit <- twostageSL(Y = train$TOTEXP, X = train[,-c(1,21)], newX = test[,-c(1,21)],
                             library.2stage = list(stage1=c("SL.glm","SL.rf.caret1","SL.glmnet"),
                                                   stage2=c("SL.logOLS.smear","SL.gammaLogGLM",
                                                            "SL.gammaIdentityGLM",
                                                            "SL.manningGLM",
                                                            "SL.gengamma","SL.coxph",
                                                            "SL.wangZhou","SL.gilleskie",
                                                            "SL.rf.caret1","SL.glmnet")),
                             library.1stage = c("SL.mean","SL.lm","SL.zip","SL.zinb","SL.tobit",
                                                "SL.tweedie","SL.rf.caret1","SL.glmnet"),
                             twostage = TRUE,
                             family.1 = binomial,
                             family.2 = gaussian,
                             family.single = gaussian,
                             cvControl = list(V = 10))
  
  # construct one-stage superlearner
  # extract onestage matrix z1
  z1 <- twostage.fit$Z[,31:38]
  onestagename <- colnames(twostage.fit$library.predict[,31:38])
  # get optimum weights for each algorithm in one-stage
  getCoef <- method.CC_LS.scale()$computeCoef(Z=z1,Y=train$TOTEXP,libraryNames=onestagename,
                                              verbose=FALSE)
  coef.onestage <- getCoef$coef
  # Prediction for each algorithm in one-stage superlearner
  predY.onestage = twostage.fit$library.predict[,31:38]
  # Compute onestage superlearner predictions on newX.
  onestage.pred <- twostage.fit$method$computePred(predY = predY.onestage, coef = coef.onestage, 
                                                   control = twostage.fit$control)
  # get discrete two-stage superlearner
  discrete.pred <- twostage.fit$library.predict[,which.min(twostage.fit$cvRisk)]
  
  ## get prediction performance
  # MSE
  mse <- c(apply(twostage.fit$library.predict, 2, function(x) mean((test$TOTEXP-x)^2)),
           mean((test$TOTEXP - onestage.pred)^2),
           mean((test$TOTEXP - twostage.fit$SL.predict)^2),
           mean((test$TOTEXP - discrete.pred)^2))
  # MAE
  mae <- c(apply(twostage.fit$library.predict, 2, function(x) mean(abs(test$TOTEXP-x))),
           mean(abs(test$TOTEXP - onestage.pred)),
           mean(abs(test$TOTEXP - twostage.fit$SL.predict)),
           mean(abs(test$TOTEXP - discrete.pred)))
  # R^2
  Rsq <-  1 - mse/var(test$TOTEXP)
  
  # algorithm name
  algo.name <- c( "S1: SL.glm + S2: SL.logOLS.smear",           
                  "S1: SL.glm + S2: SL.gammaLogGLM",            
                  "S1: SL.glm + S2: SL.gammaIdentityGLM",       
                  "S1: SL.glm + S2: SL.manningGLM",             
                  "S1: SL.glm + S2: SL.gengamma",               
                  "S1: SL.glm + S2: SL.coxph",                  
                  "S1: SL.glm + S2: SL.wangZhou",               
                  "S1: SL.glm + S2: SL.gilleskie",              
                  "S1: SL.glm + S2: SL.rf.caret1",             
                  "S1: SL.glm + S2: SL.glmnet",             
                  "S1: SL.rf.caret1 + S2: SL.logOLS.smear",     
                  "S1: SL.rf.caret1 + S2: SL.gammaLogGLM",     
                  "S1: SL.rf.caret1 + S2: SL.gammaIdentityGLM", 
                  "S1: SL.rf.caret1 + S2: SL.manningGLM",       
                  "S1: SL.rf.caret1 + S2: SL.gengamma",         
                  "S1: SL.rf.caret1 + S2: SL.coxph",            
                  "S1: SL.rf.caret1 + S2: SL.wangZhou",         
                  "S1: SL.rf.caret1 + S2: SL.gilleskie",        
                  "S1: SL.rf.caret1 + S2: SL.rf.caret1",        
                  "S1: SL.rf.caret1 + S2: SL.glmnet",       
                  "S1: SL.glmnet + S2: SL.logOLS.smear",    
                  "S1: SL.glmnet + S2: SL.gammaLogGLM",     
                  "S1: SL.glmnet + S2: SL.gammaIdentityGLM",
                  "S1: SL.glmnet + S2: SL.manningGLM",      
                  "S1: SL.glmnet + S2: SL.gengamma",        
                  "S1: SL.glmnet + S2: SL.coxph",           
                  "S1: SL.glmnet + S2: SL.wangZhou",        
                  "S1: SL.glmnet + S2: SL.gilleskie",       
                  "S1: SL.glmnet + S2: SL.rf.caret1",       
                  "S1: SL.glmnet + S2: SL.glmnet",      
                  "Single: SL.mean",                                
                  "Single: SL.lm",                                  
                  "Single: SL.zip",                                 
                  "Single: SL.zinb",                                
                  "Single: SL.tobit",                               
                  "Single: SL.tweedie",                             
                  "Single: SL.rf.caret1",                           
                  "Single: SL.glmnet",
                  "One-stage SuperLearner",
                  "Two-stage SuperLearner",
                  "Discrete SuperLearner")
  algo.num = length(algo.name)
  
  # Combine algorithm name with evaluation metrics
  # MSE
  mse_result <- data.frame("Algorithm"=algo.name,"MSE"=mse)
  # MAE
  mae_result <- data.frame("Algorithm"=algo.name,"MSE"=mae)
  # R square
  Rsq_result <- data.frame("Algorithm"=algo.name,"MSE"=Rsq)
  
  
}, error = function(e) {
  message("Caught error: ", e$message)
}, finally = {
  stopCluster(cl)
})

