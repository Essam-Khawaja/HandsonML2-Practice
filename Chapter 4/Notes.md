# Training Models

Bye-bye black boxes. Now, we delve into the depths of the machine learning models, how they work and why they work.

## Linear Regression Models
There are two ways for these models to work:
1) Using a "closed-form" equation that directly computes the model's parameters that best fit the model to the training set
2) Using an iterative optimization approach called gradient descent (GD) that gradually tweaks the model parameters to minimize the cost function over the training set

### Closed-Form Equation
Example:

y-hat = theta_0 + theta_1*x1 + theta_2*x2 + ... + theta_n * x_n

You can also alternatively do the vector representation:
y-hat = theta * x,
where x is a row vector, and theta is a column vector. Both vectors start with the 0th value.

Now, the real problem with this is to figure out the theta values. We need to find a measure of how well (or how poor) the model fits. One such measure is the RMSE (Root Mean Squared Error), which is the squared difference between the calculated target and the actual target. For RMSE, we want to minimize it, but usually in practice we minimize the MSE (Mean Squared Error), both kinda lead to the same thing.

MSE(X, h_theta) = 1/m_i * sum of ((theta * x_i) - y_i)^2

Now, in order to minimize the MSE and errors, we have another closed form equation that we can use. It is called **The Normal Equation:**

theta_hat = (X_T * X)^(-1) * X_T * y, where X is the vector of all calculated targets, where y is the vector of target variables.

Often, when we're going through finding a model, we have to deal with the noise. Even if y = 4 + 3x_1, where theta_0=4 and theta_1=3, often the noise within the dataset will not give us this exact output. The effect depends on the amount of noise, and how small the dataset is.

Now the problem with this, is that this is terrible for time complexity (or computational complexity). It usually takes about O(n^2.4) to O(n^3), usually depending on the implementation. This is usually mostly from the calculation of X_T * X, which is an (n + 1) * (n + 1) matrix. This is very bad time complexity, so instead we use *Singular Value Decomposition (SVD)*, which lowers it to O(n^2).

**Gradient Descent:**
Now forget the previous Normal Equation cause no one really does that shit lets be fr. No, instead, this is usually how most algorithms move.

Gradient Descent is a generic optimization algorithm capable of finding optimal solutions to a wide range of problems. Think of it like this:

- Plot a graph between the cost (measure of error) of a model and the theta parameters
- The graph will be parabola-like (or at least curved), where it will start high, go down to the minimum, and then go back up (maybe repeatedly)
- Hence, why do we not just take a step down the 'slope' (gradient) of the curve, until there is no slope (gradient = 0)
- Then, we know we found the minimum cost there is for the current model, and the parameters that go with it

Boom. Instead of all that matrix multiplication, we just find the best cost function through gradual steps of a gradient descent.
Now, the length of the 'baby step' that the function takes each time it tries to find the minimum point (by finding gradient = 0), is a hyperparameter called the *learning rate*. If the learning rate is too small then it take too many iterations to reach minimum, otherwise if the learning rate is too high, we would end up oscillating around the minimum instead. 

This works for one theta parameter, so how do we do it for all the parameters?
**Batch Gradient Descent:**
*Insert the partial derivative formula*

And how do we calculate the next step during a gradient descent? Using this formula:

theta_(next) = theta = eta * delta(MSE(theta))

Quick code for this algorithm:
```python
eta = 0.1
epochs = 1000
m = len(X_b)    # Number of instances

np.random.seed(42)
theta = np.random.randn(2, 1)   # Randomly initialize the theta

for currEpoch in range(epochs):
    gradients = (2 / m) * X_b.T @ (X_b @ theta - y)     # '@' means matrix multiplication
    theta = theta - eta * gradients
```

Ok, so the rate at which we take these steps is the *learning rate (eta)*, and the rate at which we find the minimum is called the *Rate of Convergence*. As you can probably see, its gonna be hard to find the right learning rate that isn't too small or too large.

Now, while this is faster than the matrix multiplication stuff from before, it is still slow. Hence, lets look at an alternative method.

**Stochastic Gradient Descent:**

