# student marks Prediction using Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = pd.read_csv("/content/Student_Marks.csv")

df = pd.DataFrame(data)

X = df[['time_study']] 
y = df['Marks']       

model = LinearRegression()
model.fit(X, y)


study_time_to_predict = np.array([[5]]) 
predicted_marks = model.predict(study_time_to_predict)

print("Predicted student marks for 5 hours of study:", predicted_marks[0])


plt.scatter(X, y, color='blue', label='Actual Marks')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("Time Study (hours)")
plt.ylabel("Student Marks")
plt.title("Student Marks Prediction using Linear Regression")
plt.legend()
plt.show()
