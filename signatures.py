#%% Libraries
import numpy as np
import numpy.random as npr
from scipy.stats import norm
import pandas as pd
import scipy.integrate as integrate
import iisignature as sig
import csv
#from esig import tosig as ts

T = 1 # one year
Sample_size = 100 # number of points in GBM
X0 = 1
d = 1 # d paths of d BM
sigma = 0.2
r = 0.02
K=1

# #%% European call option pricing & $E_n(t) = exp\left\{\left(nr+n(n-1)\frac{σ^2}{2} \right)t\right\}

def E(n,t):
    y = np.exp((n*r+0.5*n*(n-1)*sigma**2)*t)
    return y

def EC(r,X0,sigma,T,K):
    # True price by Balck-Scholes formula 
    d1= 1./(sigma*np.sqrt(T))*(np.log(X0/K)+(r+sigma**2/2)*T)
    d2= 1./(sigma*np.sqrt(T))*(np.log(X0/K)+(r-sigma**2/2)*T)
    True_price= X0*norm.cdf(d1) -K*np.exp(-r*T)*norm.cdf(d2)
    return True_price

def FC(r,X0,T,K):
  y = X0 -K*np.exp(-r*T)
  return y

def Asian_call_MC_BS(r,S0,sigma,T,K,m,n):
    
    delta=float(T/n)

    G=npr.normal(0,1,size=(m,n))

    #Log returns
    LR=(r-0.5*sigma**2)*delta+np.sqrt(delta)*sigma*G
    # concatenate with log(S0)
    LR=np.concatenate((np.log(S0)*np.ones((m,1)),LR),axis=1)
    # cumsum horizontally (axis=1)
    LR=np.cumsum(LR,axis=1)
    Spaths=np.exp(LR)
    Spaths=Spaths[:,0:len(Spaths[0,:])-1]
    #take the average over each row
    Sbar=np.mean(Spaths,axis=1)
    payoff=np.exp(-r*T)*np.maximum(Sbar-K,0) #call function

    Asian_MC_price=np.mean(payoff)

    return Asian_MC_price
m = 4 #level of signature
dim_signature = int((3**(m+1)-1)/2)
Regression_sample_size = 1000
y_EC=np.zeros((Regression_sample_size,1)) # EUROPEAN CALL OPTION
y_AC=np.zeros((Regression_sample_size,1)) # ASIAN CALL OPTION
y_LC=np.zeros((Regression_sample_size,1)) # LOOKBACK CALL OPTION WITH FLOATING STRIKE
y_FC=np.zeros((Regression_sample_size,1)) # FORWARD CONTRACT

#y_AP=np.zeros((Regression_sample_size,1)) # AMERICAN PUT
#y_VS=np.zeros((Regression_sample_size,1)) # VARIANCE SWAP

x= np.ones((Regression_sample_size,dim_signature))

t = np.zeros((1,Sample_size+1))
t[0]=np.linspace(0,1,Sample_size+1)

for i in range(Regression_sample_size):
    G = np.sqrt(T/Sample_size) * npr.normal(size=(d,Sample_size)) 
    GBM = np.concatenate((np.zeros((d,1)), np.cumsum(G,axis=1)),axis=1)
    
    BS_price = X0*np.exp((r-sigma**2/2)*t+sigma*GBM)
    augmented_path_BS = np.transpose(np.concatenate((t , BS_price , X0/T*t),axis=0))
    signature = sig.sig(augmented_path_BS,m)
    x[i,1:]= signature
    
    y_EC[i,0] = np.maximum(BS_price[0,-1]-K,0) # European Call Option
    y_LC[i,0] = np.maximum(BS_price[0,-1]-BS_price.min(),0) # Lookback Call option with floating strike
    y_AC[i,0] = np.maximum(np.sum(BS_price)/(Sample_size+1) - K,0) #Asian Call Option
    y_FC[i,0] = BS_price[0,-1]-K  # Forward Contract
    

#%% Computing expected signature
phi_0= lambda t : 1 #initial term

phi_1=[0]*3  #first order term
phi_2=[0]*9  #second order term
phi_3 =[0]*27
phi_4 = [0]*81

phi_T =np.zeros((1,dim_signature)) # phi computed t= T

phi1_1 = lambda t : t
phi1_2 = lambda t : X0*(np.exp(r*t)-1)
phi1_3 = lambda t : X0*t/T

phi_1 = [phi1_1,phi1_2,phi1_3]


for i in range(1,4):
    for j in range(1,4) :
        alpha = (i==2) +(i==3) + (j==2) +(j==3)
        ind = j-1+3*(i-1)
        
        if i==1 :
            result = lambda t, alpha=alpha, j=j : integrate.quad(lambda s : E(alpha,s)*phi_1[j-1](t-s),0,t)[0]
        if i == 2 and j==1 :        
            result = lambda t, alpha=alpha, j=j : integrate.quad(lambda s :X0*(r+sigma**2*(alpha-1))*E(alpha,s)*phi_1[1-1](t-s),0,t)[0]
        if  i==2 and j ==2 :
            result = lambda t, alpha=alpha, j=j : integrate.quad(lambda s :X0*(r+sigma**2*(alpha-1))*E(alpha,s)*phi_1[2-1](t-s)+0.5*(sigma*X0)**2*E(alpha,s)*phi_0(t-s),0,t)[0]
        if i == 3 :
            result = lambda t, alpha=alpha, j=j : integrate.quad(lambda s : X0/T*E(alpha-1,s)*phi_1[j-1](t-s),0,t)[0]
        if not ((i==1) or (i == 2 and j==1) or (i==2 and j ==2) or (i == 3)) :
            result= lambda t : 0
            
        phi_2[ind] = result
        
for i in range(1,4):
    for j in range(1,4) :
        for k in range(1,4):
            ind = k-1+3*(j-1)+9*(i-1)
            alpha = (i==2) +(i==3) + (j==2) +(j==3) + (k==2) +(k==3)
            
            if i==1 :
                result = lambda t ,alpha=alpha, j=j, k=k: integrate.quad(lambda s : E(alpha,s)*phi_2[k-1+3*(j-1)](t-s),0,t)[0]
                                
            if i == 2 and j==1 :
                result = lambda t ,alpha=alpha, j=j, k=k: integrate.quad(lambda s :X0*(r+sigma**2*(alpha-1))*E(alpha,s)*phi_2[k-1+3*(1-1)](t-s),0,t)[0]
                
            if i==2 and j ==2 :
                result = lambda t ,alpha=alpha, j=j, k=k: integrate.quad(lambda s :X0*(r+sigma**2*(alpha-1))*E(alpha,s)*phi_2[k-1+3*(2-1)](t-s)+0.5*(sigma*X0)**2*E(alpha,s)*phi_1[k-1](t-s),0,t)[0]
            
            if i == 3 :
                result = lambda t ,alpha=alpha, j=j, k=k: integrate.quad(lambda s : X0/T*E(alpha-1,s)*phi_2[k-1+3*(j-1)](t-s),0,t)[0]
            
            if not ((i==1) or (i == 2 and j==1) or (i==2 and j ==2) or (i == 3)) :
                result= lambda t : 0
            
            phi_3[ind] = result
            
for i in range (1,4):
    for j in range(1,4):
        for k in range(1,4):
            for p in range(1,4):
                ind = p-1+3*(k-1)+9*(j-1)+27*(i-1)
                alpha = (i==2) +(i==3) + (j==2) +(j==3) + (k==2) +(k==3) + (p==2) +(p==3)
                
                if i ==1 : 
                    result = lambda t ,alpha=alpha, j=j, k=k, p=p: integrate.quad(lambda s : E(alpha,s)*phi_3[p-1+3*(k-1)+9*(j-1)](t-s),0,t)[0]
                
                if i == 2 and j==1 :
                    result = lambda t ,alpha=alpha, j=j, k=k, p=p: integrate.quad(lambda s :X0*(r+sigma**2*(alpha-1))*E(alpha,s)*phi_3[p-1+3*(k-1)+9*(1-1)](t-s),0,t)[0]
                
                if i==2 and j ==2 :
                    result = lambda t ,alpha=alpha, j=j, k=k, p=p: integrate.quad(lambda s :X0*(r+sigma**2*(alpha-1))*E(alpha,s)*phi_3[p-1+3*(k-1)+9*(2-1)](t-s)+0.5*(sigma*X0)**2*E(alpha,s)*phi_2[p-1+3*(k-1)](t-s),0,t)[0]
                
                if i == 3 :
                    result = lambda t ,alpha=alpha, j=j, k=k, p=p: integrate.quad(lambda s : X0/T*E(alpha-1,s)*phi_3[p-1+3*(k-1)+9*(j-1)](t-s),0,t)[0]
                if not ((i==1) or (i == 2 and j==1) or (i==2 and j ==2) or (i == 3)) :
                    result= lambda t : 0
                phi_4[ind] = result
# Computing expected signature for t = T
phi_T[0][0]=1
a=1
for i in range(3):
    phi_T[0][a] = phi_1[i](T)
    a = a+1
    
for i in range(9) :
        phi_T[0][a] = phi_2[i](T)
        a=a+1
        
for i in range(27):
            phi_T[0][a] = phi_3[i](T)
            a=a+1

for i in range(81):
            phi_T[0][a] = phi_4[i](T)
            a=a+1        


