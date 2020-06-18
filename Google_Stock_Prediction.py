# import Libraries
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# Load Data
df_train = pd.read_csv('StockPrice_train.csv')

days = list()
adj_close = list()

df_days = df_train.loc[:, 'Date']
df_adj_close = df_train.loc[:, 'Adj Close']

for i in df_days:
    days.append([int(i.split('-')[1])])

for j in df_adj_close:
    adj_close.append(float(j))

# Create Models ( Support Vector Machine )

lin_svr = SVR(kernel='linear', C=1000.0)
lin_svr.fit(days, adj_close)

poly_svr = SVR(kernel='poly', C=1000.0, degree=2)
poly_svr.fit(days, adj_close)

rbf_svr = SVR(kernel='rbf', C=1000.0, gamma=0.15)
rbf_svr.fit(days, adj_close)

# Visualization
"""
plt.figure(figsize=(20,10))
plt.scatter(days, adj_close, color='black', label='Real Value')
plt.plot(days, rbf_svr.predict(days), color='Blue', label='RBF Model')
plt.plot(days, poly_svr.predict(days), color='Red', label='Polynomial Model')
plt.plot(days, lin_svr.predict(days), color='Purple', label='Linear Model')
plt.legend()
plt.savefig('StockPrice_Prediction.png')
plt.show()
"""


day = [[250]]

print('RBF Prediction :', rbf_svr.predict(day))
print('Linear Prediction :', lin_svr.predict(day))
print('Polynomial Prediction :', poly_svr.predict(day))
print('Real Value Price :', df_train['Adj Close'][250])