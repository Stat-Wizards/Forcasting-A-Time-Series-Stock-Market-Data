#We apply auto arima to the dataset 
modelfit <-auto.arima(GSPC$GSPC.Close, lambda = "auto")
summary(modelfit)

## Arima Results

# Diagnostics on Residuals
plot(resid(modelfit),ylab="Residuals",main="Residuals(Arima(5,1,2)) vs. Time")
plot(forecast(modelfit,h=30))

# Histogram of Residuals & Normality Assumption
hist(resid(modelfit),freq=F,ylim=c(0,9500),main="Histogram of Residuals")
e=resid(modelfit)
curve(dnorm(x, mean=mean(e), sd=sd(e)), add=TRUE, col="darkred")

# Diagnostics for Arima
tsdiag(modelfit)


# Box test for lag=2
Box.test(modelfit$residuals, lag= 2, type="Ljung-Box")
Box.test(modelfit$residuals, type="Ljung-Box")
plot(as.ts(GSPC$GSPC.Close))
lines(modelfit$fitted,col="red")

#Dataset forecasting  for the  next  30  days
price_forecast <- forecast(modelfit,h=30)
plot(price_forecast)
head(price_forecast$mean)
head(price_forecast$upper)
head(price_forecast$lower)

#Dividing the data into train & test sets , applying the model
N = length (GSPC$GSPC.Close)
n = 0.8*N
train = GSPC$GSPC.Close[1:n, ]
test = GSPC$GSPC.Close[(n+1):N,]
trainarimafit <- auto.arima(train$GSPC.Close ,lambda= "auto")
summary(trainarimafit)
predlen= length(test)
trainarima_fit <- forecast(trainarimafit, h= predlen)

#Plotting mean predicted  values vs real data
meanvalues<- as.vector(trainarima_fit$mean)
precios <- as.vector(test$GSPC.Close)
plot(meanvalues, type = "l",col="red")
lines(precios, type = "l")
#dev.off()