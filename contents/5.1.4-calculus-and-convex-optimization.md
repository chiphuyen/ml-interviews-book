#### 5.1.4 Calculus and convex optimization

1. Differentiable functions
    1. [E] What does it mean when a function is differentiable?
    1. [E] Give an example of when a function doesn’t have a derivative at a point.
    1. [M] Give an example of non-differentiable functions that are frequently used in machine learning. How do we do backpropagation if those functions aren’t differentiable?
2. Convexity
    1. [E] What does it mean for a function to be convex or concave? Draw it.
    1. [E] Why is convexity desirable in an optimization problem?
    1. [M] Show that the cross-entropy loss function is convex.
3. Given a logistic discriminant classifier:
    
    $$
        p(y=1|x) = \sigma (w^Tx)
    $$
    
    where the sigmoid function is given by:

    $$
        \sigma(z) = (1 + \exp(-z))^{-1}
    $$

    The logistic loss for a training sample $$x_i$$ with class label $$y_i$$ is given by:
    
    $$
        L(y_i, x_i;w) = -\log p(y_i|x_i)
    $$

    1. Show that $$p(y=-1|x) = \sigma(-w^Tx)$$.
    1. Show that $$\Delta_wL(y_i, x_i; w) = -y_i(1-p(y_i|x_i))x_i$$.
    1. Show that $$\Delta_wL(y_i, x_i; w)$$ is convex.

4. Most ML algorithms we use nowadays use first-order derivatives (gradients) to construct the next training iteration.
    1. [E] How can we use second-order derivatives for training models?
    1. [M] Pros and cons of second-order optimization.
    1. [M] Why don’t we see more second-order optimization in practice?
5. [M] How can we use the Hessian (second derivative matrix) to test for critical points? 
6. [E] Jensen’s inequality forms the basis for many algorithms for probabilistic inference, including Expectation-Maximization and variational inference.. Explain what Jensen’s inequality is.
7. [E] Explain the chain rule.
8. [M] Let $$x \in R_n$$, $$L = crossentropy(softmax(x), y)$$ in which $$y$$ is a one-hot vector. Take the derivative of $$L$$ with respect to $$x$$.
9. [M] Given the function $$f(x, y) = 4x^2 - y$$ with the constraint $$x^2 + y^2 =1$$. Find the function’s maximum and minimum values.

----
> On convex optimization

Convex optimization is important because it's the only type of optimization that we more or less understand. Some might argue that since many of the common objective functions in deep learning aren't convex, we don't need to know about convex optimization. However, even when the functions aren't convex, analyzing them as if they were convex often gives us meaningful bounds. If an algorithm doesn't work assuming that a loss function is convex, it definitely doesn't work when the loss function is non-convex.

Convexity is the exception, not the rule. If you're asked whether a function is convex and it isn't already in the list of commonly known convex functions, there's a good chance that it isn't convex. If you want to learn about convex optimization, check out [Stephen Boyd's textbook](http://cs229.stanford.edu/section/cs229-cvxopt.pdf).

----
> On Hessian matrix

The Hessian matrix or Hessian is a square matrix of second-order partial derivatives of a scalar-valued function. 

Given a function $$f : ℝn → ℝ$$. If all second partial derivatives of f exist and are continuous over the domain of the function, then the Hessian matrix H of f is a square nn matrix such that: $$H_{ij}=\frac{\delta f}{\delta x_i\delta x_j}$$.

<center>
    <img src="images/image18.png" width="40%" alt="Hessian matrix" title="image_tooltip">
</center>

The Hessian is used for large-scale optimization problems within Newton-type methods and quasi-Newton methods. It is also commonly used for expressing image processing operators in image processing and computer vision for tasks such as blob detection and multi-scale signal representation.