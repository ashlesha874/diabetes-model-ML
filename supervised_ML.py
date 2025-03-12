import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#Loading the diabetes dataset to a pandas DataFrame

df=pd.read_csv("diabetes.csv")
# print(df)

# #number of rows and columns in this dataset
rows_columns=df.shape
# print(rows_columns)

#getting the statistical measures of the data
statistical_measure=df.describe()
# print(statistical_measure)
outcome_number=df['Outcome'].value_counts()
# print(outcome_number)
group_by=df.groupby('Outcome').mean()
# print(group_by)

#separating the data and labels

X=df.drop(columns= 'Outcome',axis=1)
Y=df['Outcome']
# print(X)
# print(Y)

#Data standardization

scaler=StandardScaler()
scaler.fit(X)
standardized_data=scaler.transform(X)
# print(standardized_data)

X=standardized_data
Y=df['Outcome']
# print(X)
# print(Y)

#Train_Test Split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
# print(X.shape,X_train.shape,X_test.shape)

#Training the Model
classifier=svm.SVC(kernel='linear')
print(classifier)
# print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
classifier.fit(X_train,Y_train)
X_train_prediction = classifier.predict(X_train)
Y_train_accuracy = accuracy_score(Y_train,X_train_prediction)
print(Y_train_accuracy)
print(X_train_prediction)

#test data
X_test_prediction = classifier.predict(X_test)
Y_test_accuracy = accuracy_score(Y_test,X_test_prediction)
print(Y_test_accuracy)
print(X_test_prediction)

#Making a predictive system

input_data=(8,183,64,0,0,23.3,0.672,32)
input_data_as_numpy=np.array(input_data)
input_data_reshaped=input_data_as_numpy.reshape(1,-1)
std_data=scaler.transform(input_data_reshaped)
prediction=classifier.predict(std_data)
print(prediction)

if (prediction[0]==0):
    print("The patient is not diabetic")
else:
    print("the patient is diabetic")





