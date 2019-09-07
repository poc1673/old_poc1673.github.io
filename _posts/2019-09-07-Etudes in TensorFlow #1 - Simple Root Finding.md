
# Etudes with TensorFlow 1: Simple Root Finding

## Goal:

The purpose of these notebooks is to act as a source of notes and examples that I can use when implementing TensorFlow. The aim of this first notebook is to:
    
1. Explore the basics of defining variables and functions in a TensorFlow context.
2. Note basic advantages and motivations for TensorFlow.
3. Implement a simple gradient descent algorithm.

The sources are fairly simple: [Geron's book](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) on machine learning in Python, and the [TensorFlow docs](https://www.tensorflow.org/guide).


## What are the basic principles of a TensorFlow framework?

### What is a Graph?

Generally, when we think of running a program, we think of running all of the code at once. In TensorFlow, each step of calculation is broken down into specific **nodes** where a calculation is evaluated. For instance, if we want to add 2 to 8, we define one node which initializes a value to 2, and another node initialized to 8, and a third node which carries out the calculation of the two nodes that were initialized. We can also use functions from Python within the TensorFlow context.

This reductions of computations to their most basic components allows the nodes to be evaluated in parallel. Furthermore, TensorFlow performs these computations at the level of C++ which makes the code even more efficient.

Another advantages of TensorFlow is that it allows you to visualize the computation nodes that are being used. 

Implementing something in TensorFlow occurs in two steps:
1. We define the different nodes in the TensorFlow graphs.
2. We run the nodes from within a TensorFlow session.


Consider the code provided below. In it, we do the following:
    
1. Define a variable names "x" to be 3. This is done with the tf.Variable function which constructs an instance of class "Variable" within the tensorflow graph.
2. Perform the same for "y".
3. Define a function "f" with the form below. 
4. Initialize the variables using the method ".initializer" for x and y. This sets the values to be x/y.
5. Run the tensorflow session for the function "f" - this is the result that is printed.


```python
import tensorflow as tf
import numpy as np
x = tf.Variable(3, name = "x")
y = tf.Variable(4,name = "y")
f = x*x*y +y+2

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
```

    42


An alternative to the above is using the with command to run the session within a block:


```python
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    
print(result)
```

    42


### Gradient Descent

A simple example of TensorFlow is below. It's a simple root-finding algorithm being used to find the value of "f" in the code chunk above.

The steps for implementing gradient descent are as follows:
1. Define the function we are trying to optimize ("func_to_opt" in the code below).
2. Create a functional used to define how close we are to the goal. In the code below, **z** runs the function we are trying to minimize, and **abs_z** takes the square of the value.
3. Define the gradient. This is done under **gradients** where the TensorFlow function tf.gradients is used to calculate a vector of gradients.
4. Implement the gradient descent step as **training_op** where the gradients that are calculare are scaled down by the constant **steps**.

From here, we simply implement a loop where we repeatedly call the **training_op**. In this case, I call it 10,000 times before stopping. This would not be good practice, but for expediency, I've implemented it this way.

Also, note the use of the function tf.global_variables_initializer(). This simply initializes all of the variables in the graph instead of doing it one-by-one as in the earlier example.


```python
steps = .001
factors = tf.Variable(tf.random_uniform([2,1],seed = 13), name = "factors")

def func_to_opt(x,y):
    f = x*x*y +y+2
    return(f)

z = func_to_opt(factors[0],factors[1])
abs_z = pow(z,2)
gradients = tf.gradients(abs_z,factors)[0]
training_op = tf.assign(factors,factors-steps*gradients)
init = tf.global_variables_initializer()

```


```python
with tf.Session() as example_sess:
    example_sess.run(init)
    for i in range(0,10000):
        example_sess.run(training_op)
        
    print(example_sess.run(z))
    res_x = factors.eval()[0]
    res_y = factors.eval()[1]

print(res_x)
print(res_y)
```

    [5.9604645e-06]
    [1.2079186]
    [-0.813314]

