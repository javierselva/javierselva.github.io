---
layout: post
section-type: post
has-comments: false
title: Likelihood, Hessian Matrices
category: math
tags: ["math","calculus","hessian"]
---

IT SEEMS THAT RL (AND EVERYTHING IN ML REALLY) IS ABOUT THE LOG-LIKELIHOOD OF SOMETHING. LET'S UNDERSTAND THAT BITCH

What is the Hessian matrix for in Neural Networks??  But more precisely, what is it for in Log-likelihood?

A Hessian matrix represents all possible second order partial derivatives of a function.
<a id="up1"/>
A second order derivative represents "*the rate of change of the rate of change*" of a function [[1]](#sources "Wikipedia - Second derivative").

# DOES THIS MAKE SENSE???
So, if the first order derivative is showing how does a function change (i.e., its slope at each point), the second order derivative represents how fast those changes change. In the context of training a neural network, if the derivative of the loss function with respect to the input and parameters [check] represent the direction towards areas where the error is lower, the second order derivative should represent the slope of such firs derivative (?)

In the image (from: https://www.analyzemath.com/calculus/Problems/First_second_derivative.html)

We see the function f in blue, the first derivative f' in red, and the second derivative f'' in black.

As we all know, the magnitude of the derivative represents the angle? of the slope in the original function, whereas the sign shows the direction of such inclination. That is why we use the sign to go in the direction against the slope (down in the loss valley) and the magnitude of the slope is used to influence the size of the step. Also note how the derivative is 0 in both instances where f has no slope.

In this regard, the second derivative does the same for the way in which the first derivative changes. In this case it is showing the rate of change of the first derivative. Which is linearly less steep towards 0 and then increasing again. But what does it represent with regards to the loss? In the wiki article it says that it represents the acceleration with respect to position.

<a id="sources"/>
## Sources
[1] Wikipedia - Second derivative. https://en.wikipedia.org/wiki/Second_derivative [↑↑Up↑↑.](#up1)<br/>
[2] aaah!!