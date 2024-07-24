import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load the data
df = pd.read_csv("canada_per_capita_income.csv")
print(df.head())  # Print the first few rows to check column names

# Create and train the linear regression model
reg = linear_model.LinearRegression()
reg.fit(df[['year']], df['per capita income (US$)'])

# Plotting the data
plt.scatter(df['year'], df['per capita income (US$)'], color='yellow')
plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.title("Per capita income vs year")
plt.plot(df['year'], reg.predict(df[['year']]), color='blue')
plt.show()

# Now Time for Predictions
new_x = int(input("Enter the Year: "))
prediction = reg.predict(pd.DataFrame({'year': [new_x]}))
print(f"Predicted per capita income for {new_x}: {prediction[0]}")

price = reg.coef_ * new_x + reg.intercept_
print(price)

# Load additional data from Excel
d = pd.read_excel('updated.xlsx')

# Ensure the column name for prediction matches the model input
if 'year' in d.columns:
    p = reg.predict(d[['year']])
    d['prices_per_capita_$'] = p
    d.to_csv("1st_Model.csv", index=False)
else:
    print("Error: The input Excel file must contain a 'year' column.")
