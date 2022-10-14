import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
x_train = np.array([[1.0, 2.0, 3.0, 4.0]]).T
y_train = np.array([[300.0, 500.0, 700.0, 870.0]]).T

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

# Building Xbar 
one = np.ones((x_train.shape[0], 1))
Xbar = np.concatenate((one, x_train), axis = 1)
# find w ,b 
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y_train)

b= regr.coef_[0][0]
w = regr.coef_[0][1]

tmp_f_wb = compute_model_output(x_train, w, b)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()