
#libraries
library(quantmod)
library(ggplot2)
library(forecast)
library(tseries)
library(rugarch)
library(prophet)
library(tsfknn)

#data extraction and stock market indicators
getSymbols("^GSPC",src="yahoo",from="2015-01-01",to = "2020-06-04")
date = index(GSPC)
date = as.Date(date)
head(GSPC)
chartSeries(GSPC,TA = NULL)
chartSeries(GSPC,TA=c(addVo(),addBBands(),addMACD()))

#ADF TEST 
print(adf.test(GSPC$GSPC.Close))

#Plot ACF and PACF
par(mfrow = c(1, 2))
acf(GSPC$GSPC.Close)
pacf(GSPC$GSPC.Close)
par(mfrow = c(1, 1))
#dev.off()

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


#Prophet Forecasting
#Loading time series forecasting prophet package
##Dataframe creation and model application
df <- data.frame(ds = index(GSPC),
                 y = as.numeric(GSPC[,4]))
prophet_pred = prophet(df)
future = make_future_dataframe(prophet_pred,periods=30)
fcastprophet = predict(prophet_pred,future)

#Creating train prediction dataset to compare real data
dataprediction = data.frame(fcastprophet$ds,fcastprophet$yhat)
trainlen = length(GSPC$GSPC.Close)
dataprediction = dataprediction[c(1:trainlen),]
#Visualizing train prediction vs real data
p= ggplot()+
        geom_smooth(aes(x= dataprediction$fcastprophet.ds,y=GSPC$GSPC.Close),
                    colour="blue",level=0.99,fill="#69b3a2",se=T)+
        geom_point(aes(x= dataprediction$fcastprophet.ds,y=dataprediction$fcastprophet.yhat))+
        xlab("ds")+
        ylab("y= GSPC.Close")+
        ggtitle("Training Prediction vs. Real Data:Prophet")
p
#Creating Cross Validation
accuracy(dataprediction$fcastprophet.yhat,df$y)
prophet_plot_components(prophet_pred,fcastprophet)
#KNN regression Time Series Forcasting 
#Loading time series forecasting nearest neighbors package

#install.packages("tsfknn")
#Dataframe creation and model application
df <- data.frame(ds = index(GSPC),
                 y = as.numeric(GSPC[,4]))

predknn <- knn_forecasting(df$y, h = 30, lags = 1:30, k = 50, msas = "MIMO")

#Train set model accuracy
ro <- rolling_origin(predknn)
print(ro$global_accu)
autoplot(predknn)

#Fitting nnetar
lambda = BoxCox.lambda(GSPC$GSPC.Close)
dnn_fit = nnetar(GSPC[,4],lambda=lambda)
dnn_fit
fcast = forecast(dnn_fit,PI=T,h=30)
autoplot(fcast)
accuracy(dnn_fit)
