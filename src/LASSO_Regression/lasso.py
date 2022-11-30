import numpy as np
import matplotlib.pyplot as plt
# from sklearn.linear_model import SGDRegressor
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV

def load_house_data():
    data = np.loadtxt("../Gradient_Descent/data/houses.txt", delimiter=',', skiprows=1)
    X = data[:,:4]
    y = data[:,4]
    return X, y

X, y = load_house_data()
features = ['size(sqft)','bedrooms','floors','age']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle= False)

# LASSO
reg_lasso = Lasso(alpha = 1.0)
reg_lasso.fit(X, y)

# Sai số huấn luyện trên tập train
print(f"score: {reg_lasso.score(X, y)}")

# Hệ số hồi qui w và hệ số chặn b
print(f"w: {reg_lasso.coef_}") 
print(f"b: {reg_lasso.intercept_}")
#===============================================#

# tunning α with LassoCV
print("---------------LASSOCV---------------")
reg_lasso_cv = LassoCV(cv=5, random_state=0).fit(X, y)
print(f"w with tunning: {reg_lasso_cv.coef_}") 
print(f"b with tunning: {reg_lasso_cv.intercept_}")
print(f" alpha:  {reg_lasso_cv.alpha_}")
#============= predict Lasso and lassoCV
print("-------------Predict---------------")
y_lasso_pred = reg_lasso.predict(X[:4]) 
print(f"y lass: {y_lasso_pred}")
y_lasso_tunning_pred = reg_lasso_cv.predict(X[:4]) 
print(f"y lass tun: {y_lasso_tunning_pred}")
print(f" y: {y[:4]}")