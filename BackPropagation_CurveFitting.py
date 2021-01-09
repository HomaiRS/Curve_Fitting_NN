## In this script:: we generate some randome data, and using backpropagation, we fita curve on the data
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:16:17 2019

@author: Homai
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numpy import linalg
import numpy as np
import random
import math
import pandas as pd
import sys


def FirstLayer_ActivationFunc(V):
    return np.tanh(V)
    
def SecondLayer_ActivationFunc(V):
    return V
 
def FirstLayer_Activation_derivitive(V):
    return 1-(np.tanh(V))**2#1/(np.cosh(V))**2

def SecondLayer_Activation_derivitive(V):
    return 1

def PredictOutput_ofNet(Ainput,Weight1,Weight2):
    output = []
    for i in range(len(Ainput)):
        vecX = np.append([1], Ainput[i])
        v1 = W1.dot(vecX)
        phi_v1 = FirstLayer_ActivationFunc(v1)
        
        v2 = W2.dot(np.append([1], phi_v1))
        output.append(SecondLayer_ActivationFunc(v2))
    return output

Number_input = 300
X = np.random.uniform(0,0.2,Number_input)
V = np.random.uniform(-(1/10),(1/10),Number_input)

d_i = np.sin(20*X) + 3*X + V
#plt.scatter(X, d_i,c='blue'); 
#sys.exit()
##--------------------------- initialization part: initializing the zero step variables:
##---- Initialize weights
eta = 0.01; tol=1e-6
N=24; counter = 0
MaxIter=1000
Mean_MSE = np.ones(MaxIter)

W1 = np.random.normal(0,0.2, size=(N,2))
W2 = np.random.normal(0,0.2, size=(1,N+1))

while (Mean_MSE[counter] > tol) & (counter < MaxIter-1):
    Residual=[]
    for i in range(0,len(X)):
        y0 = X[i]
        ##---- induced field
        Bias_input_vec = np.append([1], y0)
        V1 = W1.dot(Bias_input_vec)
#        V1_1 = np.append([1],V1) # add bias to the V1
#        V2 = W2.dot(V1_1)
        ##---- output of y1 and y2
        y1 = FirstLayer_ActivationFunc(V1) # included bias in y as well (25*1 vector)
        V2 = W2.dot(np.append([1], y1))
        y2 = SecondLayer_ActivationFunc(V2)  
        ##---- Delta value obtained from computing gradient in the backpropagation process
#        delta2 = (d_i[i]- y2)* SecondLayer_Activation_derivitive(V2) ##CHANGE BACK
        delta2 = (y2- d_i[i])* SecondLayer_Activation_derivitive(V2)
        ##---- Delta1 : tricky :)
        W2_Nobias= np.delete(W2, 0, None)
        term1 = np.transpose(W2_Nobias)* delta2
        term2 = FirstLayer_Activation_derivitive(V1)
        delta1 = (term1*term2)
        an = np.reshape(Bias_input_vec, (1,2))    
        grad_E1 = -1 * np.reshape(delta1,(N,1)).dot(an)
        grad_E2 = -1 * delta2 * np.append([1], y1)
        
        W1 = W1 + eta * grad_E1
        W2 = W2 + eta * grad_E2
        
        Residual.append((d_i[i]- W2.dot(np.append([1], y1)))**2)
        
    Mean_MSE[counter] = np.mean(Residual)
    print(counter)
    if (Mean_MSE[counter]>Mean_MSE[counter-1]):
        eta = 0.2*eta
    counter = counter+1   
    
    
    
attempt=301  
YY = PredictOutput_ofNet(X,W1,W2)
fig = plt.figure(dpi=100); ax=fig.add_subplot(111)
ax.scatter(X, d_i,c='blue'); ax.scatter(X, np.asarray(YY),c='red')
leg=[]; leg.append('Data'); leg.append('Fit'); ax.legend(leg)
plt.savefig('/Users/Homai/Desktop/NN_CS_559/HW/HW4/MyFit2_'+str(attempt)+'.pdf')


fig = plt.figure(dpi=100); ax=fig.add_subplot(111)
ax.scatter(np.arange(counter), Mean_MSE[0:counter],c='blue'); ax.plot(np.arange(counter), Mean_MSE[0:counter],c='blue',alpha=0.7)
ax.set_xlabel('Number of iterations'); ax.set_ylabel('Mean MSE')
plt.savefig('/Users/Homai/Desktop/NN_CS_559/HW/HW4/Mean_MSE2_'+str(attempt)+'.pdf')

