import numpy as np
import pandas as pd

data  = pd.read_csv("breast-cancer.csv")
print(data.head())

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
x = data.drop(columns = ['diagnosis']).values
y = data['diagnosis'].values

x = (x-np.mean(x,axis=0))/np.std(x,axis=0)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=60)
x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=60)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def f_wb(x,w,b):
    return np.dot(x,w)+b

def cost_function(x,y,w,b=1e-10):
    m =len(y)
    y_pred = sigmoid(f_wb(x,w,b))
    epsilon = 1e-15
    y_pred = np.clip(y_pred,epsilon,1-epsilon)
    loss = (y*np.log(y_pred))+((1-y)*np.log(y_pred))
    return (-1/m)*np.sum(loss)

def gradient_descent(x,y,alpha=0.001,epochs=5000):
    m,n = x.shape
    w = np.zeros(n)
    b = 1e-10
    cost = []
    for i in range(epochs):
        z = f_wb(x,w,b)
        y_pred = sigmoid(z)
        error = y_pred -y

        dw = (1/m)*np.dot(x.T,(error))
        db = (1/m)*np.sum(error)
        
        w -=alpha *dw
        b -= alpha*db

        if i%100 ==0 :
            costs = cost_function(x,y,w,b)
            cost.append(costs)
            print(f"Epochs {i} , cost: {costs:.4f}")

    return  w,b,cost

def predict(x,w,b):
    return (sigmoid(f_wb(x,w,b))>=0.5).astype(int)

w,b,cost = gradient_descent(x_train,y_train,alpha=0.04,epochs=1000)

y_val_pred = predict(x_val,w,b)
val_accuracy = accuracy_score(y_val,y_val_pred)

print(f"Validation Accuracy : {val_accuracy:.4f}")

y_test_pred = predict(x_test,w,b)
test_accuracy = accuracy_score(y_test,y_test_pred)

print(f"Test Accuracy : {test_accuracy:.4f}")

import matplotlib.pylab as plt
plt.figure(figsize=(10, 7))
plt.plot(range(len(cost)), cost, label="Cost over epochs", color='b')
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.legend()
plt.grid()
plt.show()