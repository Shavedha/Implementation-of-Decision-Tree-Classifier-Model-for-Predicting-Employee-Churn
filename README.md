# EXPERIMENT 06: IMPLEMENTATION OF DECISION TREE CLASSIFIER MODEL FOR PREDICTING EMPLOYEE CHURN

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## EQUIPMENT'S REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM:
1. Import required packages and read the data file.
2. Use LabelEncoder to convert categorical data into numerical data.
3. Split data into training set and testing set,
4. Predict Y values.
5. Calculate accuracy of the model.

## PROGRAM:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SHAVEDHA.Y
Register Number: 212221230095  
*/
```

```
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing  import LabelEncoder
le=LabelEncoder()

data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
data.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])


```

## OUTPUT:
* data.head()      
![image](https://github.com/Rithigasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427256/2c2648b1-f46b-4efd-b8fa-78f4194a4349) 

* data.info()    
![image](https://github.com/Rithigasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427256/a84cdf5a-1495-47a4-9c49-40652c59f028)

* isnull().sum()  
![image](https://github.com/Rithigasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427256/dd06a646-11b4-44b7-9f5f-81f4e16729c9)

* Data value counts  
![image](https://github.com/Rithigasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427256/09cda506-a7e9-4fb7-bd6f-dc325cc6aebf)

* data.head() for salary
![image](https://github.com/Rithigasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427256/43f9f708-cc66-442b-bc00-326f78cb425a)

* x.head()
![image](https://github.com/Rithigasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427256/eaf9333e-1b3f-49ff-a205-dfdbd8260e75)

* Accuracy value  
![image](https://github.com/Rithigasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427256/d955494c-e3c4-4ff1-a1a4-d358c279845c)

* Data precision  
![image](https://github.com/Rithigasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427256/e8613db4-8772-44fc-93e6-eeecda2cd009)


## RESULT:
Thus, the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
