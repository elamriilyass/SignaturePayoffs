# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 13:31:13 2022

@author: ilyas
"""
import numpy as np
import scipy.integrate as integrate


def E(n,t,r=0.02,sigma =0.2):
    y = np.exp((n*r+0.5*n*(n-1)*sigma**2)*t)
    return y


def phi(X0,T=1,m=4,r=0.02,sigma=0.2) :  #expected signature
    dim_signature = int((3**(m+1)-1)/2)
    phi_0= lambda t : 1 #initial term

    phi_1=[0]*3  #first order term
    phi_2=[0]*9  #second order term
    phi_3 =[0]*27
    phi_4 = [0]*81

    phi_T =np.zeros(dim_signature) # phi computed t= T

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
    phi_T[0]=1
    a=1
    for i in range(3):
        phi_T[a] = phi_1[i](T)
        a = a+1
        
    for i in range(9) :
            phi_T[a] = phi_2[i](T)
            a=a+1
            
    for i in range(27):
                phi_T[a] = phi_3[i](T)
                a=a+1

    for i in range(81):
                phi_T[a] = phi_4[i](T)
                a=a+1    
    return phi_T