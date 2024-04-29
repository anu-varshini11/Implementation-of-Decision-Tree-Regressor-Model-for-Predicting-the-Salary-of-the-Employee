# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Collection and Preprocessing
2. Model Training
3. Model Evaluation
4. Model Deployment and Monitoring

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: ANU VARSHINI M B
RegisterNumber: 212223240010
*/
```
```
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Salary.csv")
data.head()
```
```
data.info()
```
```
data.isnull().sum()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Position']=le.fit_transform(data['Position'])
data.head()
```
```
x=data[['Position','Level']]
y=data['Salary']
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
```
```
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)
```
```
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_predict)
mse
```
```
r2=metrics.r2_score(y_test,y_predict)
r2
```
```
dt.predict([[5,6]])
```
## Output:
### data.head()

![alt text](<Screenshot 2024-04-06 114048.png>)

### data.info()

![alt text](<Screenshot 2024-04-06 114056.png>)

### data.isnull().sum()

![alt text](<Screenshot 2024-04-06 114102.png>)

### data.head()

![alt text](<Screenshot 2024-04-06 114108.png>)

### mse

![alt text](<Screenshot 2024-04-06 114115.png>)

### r2

![alt text](<Screenshot 2024-04-06 114120.png>)

### dt.predict

![alt text](<Screenshot 2024-04-06 114134.png>)
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
