
# Etudes in TensorFlow #2: Computing Gradients and Optimizing a Functional

## TensorFlow and Autodiff

One of the cornerstones of applied math is that we don't need to evaluate a derivative analytically - instead, we can approximate it. This saves time because we don't have to derive and implement the derivative. In the past, I've implemented a quadratic method for logistic regression which while doable, was time consuming. TensorFlow offers a nice, fast was to circumvent this.

Furthermore, differentiation may be borderline unfeasible, or even impossible in some cases. For instance, if you are trying to optimize something based off a for loop, (like the example in *Hands on Machine Learning with Scikit Learn and Tensorflow* on page 238) it may be practically impossible. However, autodiff makes this simple. 

A simple validation example comparing the analytic and autodiff results for the equation below are provided in the next two charts:

$$ z =  x^3 + y^3 +x^2*y $$

### Example of Calculating a Derivative


```python
import tensorflow as tf
import numpy as np
def f(x,y):
    return(pow(x,3) + pow(y,3) + pow(x,2)*y)

def dfx(x,y):
    return(3*pow(x,2)  + 2*pow(x,1)*y)
def dfy(x,y):
    return( 3*pow(y,2) + pow(x,2))

def grad_f(x,y):
    return([dfx(x,y),dfy(x,y)])
print(f(2.0,3.0))
print(grad_f(2.0,3.0))

```

    47.0
    [24.0, 31.0]
    

In TensorFlow, we can calculate the gradient like above using autodifferentiation. This strategy repeatedly uses the chain rule for each node in the graph to calculate the derivative. Originally, I was going to provide a simple example, but there are enough guides to it already (again, [O'Reilly](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291) but another example is provided [here](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation). 

To implement it in TensorFlow, we use a function that I also employed in the last post: **tf.gradient** which adds a node to the graph where one tensor is differentiated with respect to its inputs. See below:


```python
x = tf.Variable(2.0,name = "x")
y = tf.Variable(3.0,name = "y")

factors = tf.Variable([x,y],name = "factors")
z = f(factors[0],factors[1])
df = tf.gradients(z,factors)[0]
 
init = tf.global_variables_initializer()
with tf.Session() as test_sess:
    test_sess.run(init)
    print(df.eval())
    
```

    [24. 31.]
    

From the above, we can see that we are able to get the exact same results whether we calculate the derivative analytically, or with TensorFlow's approximation.

## Simple Linear Regression

I'll provide a simple introductory example that can be worked up from scratch: Linear regression. Before going on, I'll add that I will not assess the models - the main purpose of these notes is to review building the model using the data, the loss functional, and the optimizers.


```python
import sys
from sklearn.datasets import fetch_california_housing
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Step 1: Download and set up data:
housing = fetch_california_housing()
m, n = housing.data.shape
# Step 2: Add bias to the information:
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]

# Set up X as the data. Set up y as the dependent variable:
X = tf.constant(housing_data_plus_bias, dtype = tf.float32, name = "X")
y = tf.constant(housing.target.reshape(-1,1), dtype = tf.float32, name = "y")
XT = tf.transpose(X)

theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)), XT), y)
with tf.Session() as sess: 
    theta_value = theta.eval()
   
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

# Define loss function. This will be passed to autodiff instead of a hands-on calculation of the gradient:
def MSE_loss_fun(y_pred,y):
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    return(mse)

n_epochs = 50000 # The number of iterations for our gradient descent method along with the learning rate.
learning_rate = 0.05

# Finally: Set up the nodes for the graph being used for this linear regression.
# Note that the values for X and y are constant since we are going optimizing for the coefficients.

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
# The node where the loss function is applied is here:
mse = MSE_loss_fun(y_pred = y_pred, y = y)
# Create a node to calculate the gradient:
gradients = tf.gradients(mse, [theta])[0]
training_op = tf.assign(theta, theta - learning_rate * gradients)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 10000 == 0:
            print("Epoch", epoch,";   MSE =", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()
```

    Epoch 0 ;   MSE = 2.7544262
    Epoch 10000 ;   MSE = 0.524321
    Epoch 20000 ;   MSE = 0.524321
    Epoch 30000 ;   MSE = 0.524321
    Epoch 40000 ;   MSE = 0.524321
    


```python
best_theta
```




    array([[ 2.0685573 ],
           [ 0.8296168 ],
           [ 0.11875109],
           [-0.26552254],
           [ 0.3056927 ],
           [-0.00450319],
           [-0.03932615],
           [-0.89989173],
           [-0.8705468 ]], dtype=float32)




```python
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
reg = linear_model.LinearRegression()
features = scaled_housing_data_plus_bias
preds = housing.target.reshape(-1, 1)
```


```python
fitted_model = reg.fit(features,preds)
fitted_model.coef_
```




    array([[ 0.        ,  0.8296193 ,  0.11875165, -0.26552688,  0.30569623,
            -0.004503  , -0.03932627, -0.89988565, -0.870541  ]])



## Classification by Logistic Regression


For a more direct example of doing this, I'll perform a quick example of implementing logistic regression using gradient descent. The main piece for this will be implementing the loss functional for logistic regression:

$$ J(\theta) = -\frac{1}{m}\sum_i^N \Big[y_i*log( \frac{1}{1+Exp(-(\beta_0+\sum_i^k \beta_i x_i))} ) + (1-y^i)log(1-\frac{1}{1+Exp(-(\beta_0+\sum_i^k \beta_i x_i))})\Big]$$




```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Filter based on whether they are 0 or 1:
filter_inds = np.where((y==0)|(y==1))
y = y[filter_inds ]
y = y.reshape(-1,1)
X = X[filter_inds,:][0]
m, n = X.shape

X_model = tf.placeholder(tf.float64, shape=(None, n), name="X")
y_model = tf.placeholder(tf.float64, shape=(m, None), name="y")
coefs = tf.Variable(tf.random_uniform([n, 1], -1.0, 1.0, seed=13), name="coefs")

logits = tf.matmul( X_model, tf.cast(coefs,tf.float64), name="logits")
y_probability = tf.sigmoid(logits)
loss = tf.losses.log_loss(y_model, y_probability)  # uses epsilon = 1e-7 by default
learning_rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

n_epochs = 100000
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        sess.run(training_op, feed_dict={X_model: X, y_model: y})
        loss_val = loss.eval({X_model: X, y_model: y})
        if epoch%(n_epochs/10)==0:
            print("Epoch:", epoch, "\tLoss:", loss_val)
        y_proba_val = y_probability.eval(feed_dict={X_model: X, y_model: y})     
    return_coefs = coefs.eval()
    
print(return_coefs)
```

    Epoch: 0 	Loss: 2.3958693
    Epoch: 10000 	Loss: 6.5535634e-05
    Epoch: 20000 	Loss: 4.3094485e-07
    Epoch: 30000 	Loss: -1.1384486e-07
    Epoch: 40000 	Loss: -1.1920928e-07
    Epoch: 50000 	Loss: -1.1920928e-07
    Epoch: 60000 	Loss: -1.1920928e-07
    Epoch: 70000 	Loss: -1.1920928e-07
    Epoch: 80000 	Loss: -1.1920928e-07
    Epoch: 90000 	Loss: -1.1920928e-07
    [[-6.1189585]
     [-9.729725 ]
     [17.801567 ]
     [20.739258 ]]
    

I think I'm at the point where I want to do something more creative and train a model using TensorFlow - that will be a topic for my next TensorFlow post though.
