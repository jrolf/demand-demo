
import numpy  as np
import pandas as pd 
import datetime,pickle  
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression as OLS

######################################################################
######################################################################

def simple_rmse(x,y):
    deltas = np.array(x)-np.array(y) 
    squares = deltas**2.0
    summed = squares.sum() 
    rmse = (summed/float(len(x)))**0.5 
    return rmse 
    
def OlsFromPoints(xvals,yvals):
    LEN = len(xvals)
    xvals = np.array(xvals).reshape(LEN,1) 
    model = OLS()
    model.fit(xvals,yvals) 
    return model  

def GetOlsParams(ols_model):
    m = ols_model.coef_[0] 
    b = ols_model.intercept_
    return m,b 

# Given an OLS Model, Return an Inverse Model
def GenInverseOLS(normal_ols_model):
    m,b = GetOlsParams(normal_ols_model)
    inv_func = lambda y: (y-b)/float(m) 
    
    xvals = np.linspace(-100,100,1000)
    yvals = inv_func(xvals) 
    inv_ols_model = OlsFromPoints(xvals,yvals) 
    return inv_ols_model 


class PolyFit:
    def __init__(self,poly=[2,3,4,5]):
        if 'int' in str(type(poly)): poly=[poly]
        self.poly = poly
        self.models = {}
        
    def fit(self,x_train,y_train,poly=[]):
        if poly: 
            if 'int' in str(type(poly)): poly=[poly]
            self.poly = poly
        x = np.array(x_train)
        y = np.array(y_train)
        if x.shape == (len(x),1): x = x.reshape([len(x),]) 
        if y.shape == (len(y),1): x = x.reshape([len(y),])  
        results = []
        for deg in self.poly:
            params = np.polyfit(x,y,deg)
            self.models[deg] = params    

    def predict(self,x_test): 
        x = np.array(x_test) 
        if x.shape == (len(x),1): x = x.reshape([len(x),])
        results = []
        for deg in self.poly:
            params = self.models[deg] 
            preds = np.polyval(params,x)
            results.append(preds)
        M = np.array(results)
        preds_final = M.mean(0) 
        return preds_final


######################################################################
######################################################################







