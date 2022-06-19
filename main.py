#%% Setting up environment
from os import chdir
chdir(wd)

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pandas as pd
import iisignature as sig
import csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from expected_signature import phi 
from options_real_prices import FC, EC, AC, LC
import time


T = 1 # one year
Sample_size = 100 # number of points in GBM
X0 = 1
d = 1 # d paths of d BM
sigma = 0.2
r = 0.02
K=1
m = 4 #level of signature

Regression_sample_size = 40000
validation_size =10

dim_signature = int((3**(m+1)-1)/2)
k = np.linspace(0.6,1.7,validation_size)
x0 = 1 * k

#%% Computing expected signature
start = time.time()

real_prices = np.zeros((validation_size,1))
phi_T_database = np.zeros((validation_size,121))
for i in range(validation_size) :
    print(i)
    X0 = x0[i]
    K = k[i]
    phi_T_database[i,:] = phi(X0)
g = open("phi_T_database.csv",'w')
writer = csv.writer(g,delimiter=';')
writer.writerows(phi_T_database)
g.close()

end = time.time()
print(end - start)
#%% Computing real prices
for j in range(validation_size) :
    X0 = x0[j]
    K = k[j]
    real_prices[j] = LC(X0)
    print(j)
#%% Computing linear functionals 
t = np.zeros((1,Sample_size+1))
t[0]=np.linspace(0,1,Sample_size+1)
xxx = np.ones((validation_size,Regression_sample_size,dim_signature))
yyy = np.zeros((validation_size,Regression_sample_size,1))
x =np.zeros((validation_size*Regression_sample_size*dim_signature,1))
y = np.zeros((validation_size*Regression_sample_size,1))

start = time.time()
for j in range(validation_size) : 
    print(j)
    X0 = x0[j]
    K = k[j]    
    for i in range(Regression_sample_size):
        G = np.sqrt(T/Sample_size) * npr.normal(size=(d,Sample_size)) 
        GBM = np.concatenate((np.zeros((d,1)), np.cumsum(G,axis=1)),axis=1)
        
        BS_price = X0*np.exp((r-sigma**2/2)*t+sigma*GBM)
        augmented_path_BS = np.transpose(np.concatenate((t , BS_price , X0/T*t),axis=0))
        signature = sig.sig(augmented_path_BS,m)
        xxx[j,i,1:]= signature
        yyy[j,i,0] = BS_price[0,-1] - np.min(BS_price[0])
        #yyy[j,i,0] = np.maximum(BS_price[0,-1]-K,0) 
        #yyy[j,i,0] = np.maximum(np.mean(BS_price) - K,0)
        #yyy[j,i,0] = BS_price[0,-1]-K
        
x[:,0] = xxx.reshape(validation_size*Regression_sample_size*dim_signature)
y[:,0] = yyy.reshape(validation_size*Regression_sample_size)
end = time.time()
print(end - start)
print("x")
h = open("xxx.csv",'w')
writer = csv.writer(h,delimiter=';')
writer.writerows(x)
h.close()
print("y")
hh = open("yyy.csv",'w')
writer = csv.writer(hh,delimiter=';')
writer.writerows(y)
hh.close()

end = time.time()
print(end - start)
#%% Plotting approximated prices compared to real prices
approximated_prices = np.array(pd.read_csv ("Approximated prices.csv"))

score =r2_score(real_prices,approximated_prices)
error = np.sqrt(mean_squared_error(real_prices, approximated_prices))

plt.title("Lookback Call Option with floating strike \n $R^2$ = " + str(score) + "\n RMSE = " + str(error)+"\n Regression_sample_size = " + str(Regression_sample_size) + "\n # Market conditions = " + str(validation_size))
plt.xlabel("Real prices")
plt.ylabel("Approximated prices")
plt.grid()
plt.plot(real_prices,approximated_prices,"k--",marker = 'o',markerfacecolor="green")
plt.plot(real_prices,real_prices)
# plt.plot(real_prices-approximated_prices) 
# plt.plot(approximated_prices,"k--",marker = 'o',markerfacecolor="green")
