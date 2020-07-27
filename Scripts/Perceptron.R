#Fitting nnetar
lambda = BoxCox.lambda(GSPC$GSPC.Close)
dnn_fit = nnetar(GSPC[,4],lambda=lambda)
dnn_fit
fcast = forecast(dnn_fit,PI=T,h=30)
autoplot(fcast)
accuracy(dnn_fit)