import numpy as np
from scipy.stats import norm
import numpy.random as npr

# European Call option
def EC(X0,K,sigma = 0.2 ,T = 1,r =0.02 ):
    # True price by Balck-Scholes formula 
    d1= 1./(sigma*np.sqrt(T))*(np.log(X0/K)+(r+sigma**2/2)*T)
    d2= 1./(sigma*np.sqrt(T))*(np.log(X0/K)+(r-sigma**2/2)*T)
    True_price= X0*norm.cdf(d1) -K*np.exp(-r*T)*norm.cdf(d2)
    return True_price

# Asian Call Option 
def AC(X0,K,sigma=0.2,r=0.02,T=1,m=1000000,n=100):
    delta=float(T/n)
    G=npr.normal(0,1,size=(m,n))
    #Log returns
    LR=(r-0.5*sigma**2)*delta+np.sqrt(delta)*sigma*G
    # concatenate with log(X0)
    LR=np.concatenate((np.log(X0)*np.ones((m,1)),LR),axis=1)
    # cumsum horizontally (axis=1)
    LR=np.cumsum(LR,axis=1)
    Spaths=np.exp(LR)
    Spaths=Spaths[:,0:len(Spaths[0,:])-1]
    #take the average over each row
    Sbar=np.mean(Spaths,axis=1)
    payoff=np.exp(-r*T)*np.maximum(Sbar-K,0) #call function
    Asian_MC_price=np.mean(payoff)
    return Asian_MC_price

# Forward Contract
def FC(X0,K,r=0.02,T=1):
  y = X0 -K*np.exp(-r*T)
  return y

# Lookback Call Option with floating strike
def LC(X0,sigma=0.2,r=0.02,T=1,m=1000000,n=100) : 
    
    t = np.zeros((1,n+1))
    t[0]=np.linspace(0,1,n+1)
    G = np.sqrt(T/n) * npr.normal(size=(m,n)) 
    GBM = np.concatenate((np.zeros((m,1)), np.cumsum(G,axis=1)),axis=1)
    BS_price = X0*np.exp((r-sigma**2/2)*t+sigma*GBM)
    payoff = BS_price[:,-1] - np.min(BS_price,axis = 1)
    true_price = np.exp(-r*T)*np.mean(payoff)
    return true_price
# Spread options $S_2-S_1$
def margrabe(X0_1,X0_2,sigma_1=0.2,sigma_2=0.2, r = 0.02,T =1 ,rho = 0.5):
    sigma = np.sqrt(sigma_1**2 + sigma_2**2 - 2*rho*sigma_1*sigma_2)
    d1 = np.log(X0_2/X0_1)/(sigma*np.sqrt(T)) + 0.5*sigma*np.sqrt(T)
    d2 = d1 - sigma*np.sqrt(T)
    true_price = X0_2*norm.cdf(d1) - X0_1*norm.cdf(d2)
    return true_price

def spread_option_MC(X0_1,X0_2,sigma_1,sigma_2, r = 0.02,T =1,rho = 0.5,Sample_size=10000000):
    B_1 = np.sqrt(T)*npr.normal(0,1,size=(1,Sample_size))
    B_2 = np.sqrt(T)*npr.normal(0,1,size=(1,Sample_size))
    
    X_1 = X0_1*np.exp((r-0.5*sigma_1**2)*T+sigma_1*B_1)
    X_2 = X0_2*np.exp((r-0.5*sigma_2**2)*T+sigma_2*(rho*B_1+np.sqrt(1-rho**2)*B_2))
    
    payoff = np.maximum(X_2 - X_1,0)
    true_price =np.exp(-r*T)*np.mean(payoff)
    return true_price
    
    

    
    

    
