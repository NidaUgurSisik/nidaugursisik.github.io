---
layout: post
title: "Gradient Descent"
date: 2021-02-08
excerpt: "Machine Learning"
tags: [machine learning, gradient descent, deep learning]
---

### Gradient Descent
Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model. Parameters refer to coefficients in Linear Regression and weights in neural networks.

### Gradient Descent Variants
There are three variants of gradient descent, which differ in how much data we use to compute the gradient of the objective function. Depending on the amount of data, we make a trade-off between the accuracy of the parameter update and the time it takes to perform an update.

#### Batch Gradient Descent
Vanilla gradient descent, aka batch gradient descent, computes the gradient of the cost function w.r.t. to parameters θ for the entire training dataset:

$$θ=θ−η⋅∇_θJ\left(θ\right)$$

As we need to calculate the gradients for the whole dataset to perform just one update, batch gradient descent can be very slow and is intractable for datasets that do not fit in memory. Batch gradient descent also does not allow tus to update our model online.

```python
for i in range(nb_epochs):
    params_grad = evaluate_gradient(loss_function, data, params)
    params = params - learning_rate * params_grad
```
For a pre-defined number of epochs, we first compute the gradient vector `params_grad` of the loss function for the whole dataset w.r.t. our parameter vector `params`. Note that state-of-the-art deep learning libraries provide automatic differentiation that efficiently computes the gradient w.r.t. some parameters. If you derive the gradients yourself, then gradient checking is good idea.

We then update our parameters in the opposite direction of the gradients with the learning rate determining how big of an update we perform. Batch gradient descent is guaranteed to converge to the global minimum for convex error surfaces and to a local minimum for non-convex surfaces.

#### Stochastic Gradient Descent
Stochastic gradient descent (SGD) in constrast performs a parameter update for each training example $x^i$ and label $y^i$:

$$θ=θ−η⋅∇_θJ\left(θ;x^i;y^i\right)$$

Batch gradient descent performs redundant computations for large datasets, as it recomputes gradients for similar examples before each parameter update. SGD does away with this redundancy by performing one update at a time. It is therefore usually much faster and can be also be used to learn online.

While batch gradient descent converges to the minimum of the basin the parameters are placed in, SGD's fluctation, on the one hand, enables it to jump to new and potentially better local minima. On the other hand, this ultimaltely complicates convergence to the exact minimum, as SGD will keep overshooting. However, it has been shown that when we slowly decrease the learning rate, SGD shows the same convergence behaviour as batch gradient descent, almost certainly converging to a local or the global minimum for non-convex and convex optimization respectively. Its code fragment simply adds a loop over the training examples and evaluates the gradient w.r.t. each example.
```python
for i in range(nb_epochs):
  np.random.shuffle(data)
  for example in data:
    params_grad = evaluate_gradient(loss_function, example, params)
    params = params - learning_rate * params_grad
```

#### Mini-Batch Gradient Descent
Mini-batch gradient descent finally takes the best of both worlds and performs an update for every mini-batch of n training examples:

$$θ=θ−η⋅∇_θJ\left(θ;x^{i:i+n};y^{i:i+n}\right)$$

This way, it **a)** reduces the variance of the parameter updates, which can lead to more stable convergence; and **b)** can make use of highly optimized matrix optimizations common to state-of-the-art deep learning libraries that make computing the gradient w.r.t. a mini-batch very efficient. Common mini-batch sizes range between 50 and 256, but can vary for different applications. Mini-batch gradient descent is typically the algorithm of schoice when training a neural network and the term SGD usually is employed also when mini-batches are used. Note: In modifications of SGD in the rest of this post, we leave out the parameters $x^{i:i+n};y^{i:i+n}$ for simplicity.

In code, instead of iterating over examples, we now iterate over mini-batches of size 50;
```python
for i in range(nb_epochs):
  np.random.shuffle(data)
  for batch in get_batches(data, batch_size=50):
    params_grad = evaluate_gradient(loss_function, batch, params)
    params = params - learning_rate * params_grad
```

### REFERENCES
>https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html
>https://ruder.io/optimizing-gradient-descent/