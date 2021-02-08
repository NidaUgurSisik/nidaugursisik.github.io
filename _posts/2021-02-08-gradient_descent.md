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
>θ=θ−η⋅∇_θJ\left(θ\right)



<figure>
	<a href="https://s3-us-west-2.amazonaws.com/courses-images/wp-content/uploads/sites/1861/2017/06/23162145/re1i9ogssarmuhjlkzto.png"><img src="https://s3-us-west-2.amazonaws.com/courses-images/wp-content/uploads/sites/1861/2017/06/23162145/re1i9ogssarmuhjlkzto.png"></a>
</figure>


### REFERENCES
>https://courses.lumenlearning.com/boundless-algebra/chapter/introduction-to-matrices/