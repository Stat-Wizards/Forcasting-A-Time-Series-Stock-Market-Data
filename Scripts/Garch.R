
# Fitting GARCH

#Dataset forecast upper first 5 values
fitarfima = autoarfima(data = GSPC$GSPC.Close, ar.max = 5, 
                       ma.max = 2,criterion = "AIC", method = "full")
fitarfima$fit
#define the model
garch11closeprice=ugarchspec(variance.model=list(garchOrder=c(1,1)),
                             mean.model=list(armaOrder=c(3,2)))
#estimate model 
garch11closepricefit=ugarchfit(spec=garch11closeprice, data=GSPC$GSPC.Close)

#conditional volatility plot
plot.ts(sigma(garch11closepricefit), ylab="sigma(t)", col="blue")

#Model akike
infocriteria(garch11closepricefit)
#Normal residuals
garchres <- data.frame(residuals(garch11closepricefit))
plot(garchres$residuals.garch11closepricefit)

#Standardized residuals
garchres <- data.frame(residuals(garch11closepricefit, standardize=TRUE))
#Normal Q plot
qqnorm(garchres$residuals.garch11closepricefit,standardize=TRUE)
qqline(garchres$residuals.garch11closepricefit,standardize=TRUE)

#Squared standardized residuals Ljung Box
garchres <- data.frame(residuals(garch11closepricefit, standardize=TRUE)^2)
Box.test(garchres$residuals.garch11closepricefit..standardize...TRUE..2, type="Ljung-Box")

#GARCH Forecasting
garchforecast <- ugarchforecast(garch11closepricefit, n.ahead = 30 )
plot(garchforecast) #Selection: 1