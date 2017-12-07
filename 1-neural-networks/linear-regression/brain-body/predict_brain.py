import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

#train model on data
body_reg = LinearRegression()
body_reg.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
