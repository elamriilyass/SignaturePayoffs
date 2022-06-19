rm(list=objects())
setwd("C:/Users/ilyas/Bureau/ENSTA/Finance/PRE")
library(ramify)
r=0.02
T=1
Regression_sample_size = 40000
validation_size =10
dim_signature = 121

phi_T_database = read.csv("phi_T_database.csv", header=F,sep=";",dec=".")

yyy = as.matrix(read.csv("yyy.csv ",header = F , sep = ";", dec = "."))
yyy = resize(yyy,1,Regression_sample_size,validation_size)

xxx = as.matrix(read.csv("xxx.csv ",header = F , sep = ";", dec = "."))
xxx = resize(xxx,dim_signature,Regression_sample_size,validation_size)
approximated_prices = 1:validation_size
for (i in 1: validation_size)
{
  phi_T = phi_T_database[i,]
  y_EC = yyy[,,i]
  dataset = as.data.frame(t(xxx[,,i]))
  model_EC <- lm(y_EC ~., data = dataset)
  EC_price = exp(-r*T)*predict(model_EC, newdata = phi_T) 
  approximated_prices[i] = EC_price
}

approximated_prices = as.data.frame(approximated_prices)
write.csv(approximated_prices,"Approximated prices.csv", row.names = FALSE)

