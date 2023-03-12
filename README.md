# <p align="center">Developing a Neural Network Regression Model</p>

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. These units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

Build your training and test set from the dataset, here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model
<p align="center">
    <img width="560" alt="image" src="./NN_Model.png">
</p>

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar object, fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

### Importing Modules
```py
#Use this to connect to google drive & Access live Sheets
from google.colab import auth
import gspread
from google.auth import default

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential as Seq
from tensorflow.keras.layers import Dense as Den
from tensorflow.keras.metrics import RootMeanSquaredError as rmse
```
### Authenticate &  Create Dataframe using Data in Sheets
```py
#Authenticate
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

sheet = gc.open('Multiple').sheet1 
rows = sheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Table':'int'})
df = df.astype({'Product':'int'})
```
### Assign X and Y values
```py
x = df[["Table"]] .values
y = df[["Product"]].values
```
### Normalize the values & Split the data
```py
scaler = MinMaxScaler()
scaler.fit(x)
x_n = scaler.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_n,y,test_size = 0.3,random_state = 3)
```
### Create a Neural Network & Train it
```py
ai = Seq([
    Den(8,activation = 'relu',input_shape=[1]),
    Den(15,activation = 'relu'),
    Den(1),
])

ai.compile(optimizer = 'rmsprop',loss = 'mse')

ai.fit(x_train,y_train,epochs=2000)
ai.fit(x_train,y_train,epochs=2000)
```
### Plot the Loss
```py
loss_plot = pd.DataFrame(ai.history.history)
loss_plot.plot()
```
### Evaluate the model
```py
err = rmse()
preds = ai.predict(x_test)
err(y_test,preds)
```
### Predict for some value
```py
x_n1 = [[30]]
x_n_n = scaler.transform(x_n1)
ai.predict(x_n_n)
```
## Dataset Information

<p align="center">
    <img width="350" alt="image" src="./dataset.png">
</p>

## OUTPUT

### Training Loss Vs Iteration Plot

![Plot](./loss.png)
![Plot](./eval.png)

### Test Data Root Mean Squared Error

![RMSE](./rmse.png)

### New Sample Data Prediction

![Prediction](./pred.png)

## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully