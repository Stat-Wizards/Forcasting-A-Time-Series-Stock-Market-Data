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