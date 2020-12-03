import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)

prediction = model.predict(x_train)

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, model.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('years of exp.')
plt.ylabel('salary')
plt.show()

plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, model.predict(x_test), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('years of exp.')
plt.ylabel('salary')
plt.show()

val = input('Enter the Years of Experience: ').split(',')
for i in val:
    new_predict = model.predict([[float(i)]])
    for j in new_predict:
        print(f'The Avg. salary for the person with {i} years of experience is',j)