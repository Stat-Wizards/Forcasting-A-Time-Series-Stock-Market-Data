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