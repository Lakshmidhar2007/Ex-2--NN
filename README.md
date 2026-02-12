<H3>Name : LAKSHMIDHAR N</H3>
<H3>Register no : 212224230138</H3>
<H3>Date : 12.02.2026</H3>
<H3>Experiment No. 2 </H3>
## Implementation of Perceptron for Binary Classification
# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>

STEP 11:Print the accuracy<BR>
# PROGRAM:
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = 0
        self.errors = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, epochs=10):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            linear_output = np.dot(xi, self.weights) + self.bias
            y_pred = self.sigmoid(linear_output)
            y_pred_class = 1 if y_pred >= 0.5 else -1

            update = self.learning_rate * (yi - y_pred_class)
            self.weights += update * xi
            self.bias += update

            errors += int(update != 0)

            self.errors.append(errors)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_output)
        return np.where(y_pred >= 0.5, 1, -1)

df = pd.read_excel('/content/Iris.xlsx')


X = df.iloc[0:100, 0:2].values
y = df.iloc[0:100, 4].values

y = np.where(y == 'Iris-setsa', 1, -1)

X = X.astype(float)

X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


model = Perceptron(learning_rate=0.01)
model.fit(X_train, y_train)

plt.plot(range(1, len(model.errors) + 1),model.errors, marker='o')

plt.xlabel("Epoch")
plt.ylabel("Errors")
plt.title("Training Errors per Epoch")
plt.show()

accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
print("Accuracy:", accuracy, "%")
```


# OUTPUT:
![alt text](image-1.png)

![alt text](image.png)



![alt text](image-2.png)

![alt text](image-3.png)

# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
