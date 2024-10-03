---
layout: post
section-type: post
has-comments: false
title: Answering typical ML interview questions.
category: various
tags: ["various","interview","ML","basics"]
---

# How to check overfitting? How to deal with it?

## My initial answer: 
Overfitting occurs when a model exhibits great accuracy with training samples, but does poorly on validation or test sets. This occurs because the model is starting to memorize or “over-fit” the patterns shown by the training data set. This could be a result of an overly capable network (too big, too many weights, allowing the network for such behaviour), and reducing the number of weights (e.g., number of hidden layers, or their dimensionality) should help avoid memorization.

## My final answer:
 - Small dataset, not enough to properly find relevant patterns in the data that allow for generalization.
 - Too many epochs, allowing the network to excessively fit the training data.
 - Noisy data (e.g., too much irrelevant information) may aggravate the problem. A bigger than enough network will pick up patterns on that irrelevant information and misinterpret correlation with causality. 
 - In this case, memorization could be see as a shortcut the network is taking. When a network memorizes instead of learning, it leads to a loss of generalization abilities.
 - To check if that is indeed the case, one may have to use K-fold cross-validation. Low error rates and a high variance are good indicators of overfitting.
 - Possible solutions:
	 - Data augmentation: May help to virtually increase the size of the training data and making the model more robust.
	 - Early stopping: Stop before the network has a change to overfit. However, a fine balance has to be found. Stopping too early may not give the network a change to learn the necessary patterns in the data (underfitting).
	 - Adding data: This must be done carefully, not to add even more complexity to the problem! If data is not clean (adds irrelevant information / noise) it may make problems worse by not allowing the network to learn.
	 - Feature selection / Prunning: Detect irelevant features / parameters and remove them, to allow for relevant features to be dected.
	 - Regularization: e.g. Dropout or other methods (lasso regularization, ridge regression)... I've read this reduces noise, but I believe what it does is actually increase it! Forcing the network to become robust to it.
	 - Ensemble methods: (e.g. decision trees), train multiple classifiers on different subsets of the data in order to reduce the variance of the problem. Then find a consensus between the multiple models (bagging and boosting).

The [variance / bias trade-off](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff).

    - The bias error is an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).
    - The variance is an error from sensitivity to small fluctuations in the training set. High variance may result from an algorithm modeling the random noise in the training data (overfitting).

So, as I understand it, this is not strictly or necessarily related to the network size, it may also have to do with the quality of the data. I think, bias relates to a particular network finding inappropriate solutions and getting stuck there (local minima?), while variance means that the network is not able to settle in a particular shape/function, because it keeps changing (great loss signal) from every or many of the input samples.

Low variance, high bias: Underfitting
High variance, low bias: Overfitting.

When talking about this problem, *complexity* is not only talking about model size, it is the overall problem complexity of fitting a net to some data, meaning that complexity here relates to how hard is the data, how big is the model, etc.

There's ongoing research about overfitting, as phenomenons such as Double Descent and Grokking, which are still not fully understood.

# How does SGD / Adam / etc work?
From https://en.wikipedia.org/wiki/Stochastic_approximation, it is a form to iteratively approximate a function which cannot be done directly but can be estimated through a set of noisy observations.

From: https://en.wikipedia.org/wiki/Stochastic_gradient_descent "It can be regarded as a stochastic approximation of gradient descent optimization, since it replaces the actual gradient (calculated from the entire data set) by an estimate thereof (calculated from a randomly selected subset of the data)."

Gradient descent is the idea of going agains the sign of the derivative in order to reach local minimas (with appropriately small step size). Stochastic gradient descent is doing that for each sample instead of for the whole data at once. In practice using mini-batches works better than one at a time (allows to vectorize some operations making it faster).

I think in that article above is pretty clear. It is the classic rule of w = w + a\*d(xw - y)/dx donde a es el learning rate.

Various improvements:
	- Adding a learning rate schedule.
	- Momentum adds a compounded update such that the new update is a linear combination of all older updates with the current one. "it tends to keep traveling in the same direction, preventing oscillations". In this sense, it stabilizes training. 
	- AdaGrad. Useful for sparse data problems. Uses different learning rates for each parameter. They're adapted depending on the history of the gradients that each parameter receives. It encourages parameters that receive a many or big updates to have a lower learning rate, and vice-versa for parameters that receive few updates.
	- RMSProp. Is kinda an extension to AdaGrad. Seems to have something different in the way the mean of past gradients for each weight are stored, but " 'forgetting' is introduced to solve Adagrad's diminishing learning rates in non-convex problems by gradually decreasing the influence of old data".
	- Adam = RMSProp with Momentum. It includes a couple of forgetting parameters. Adam does not converge for all convex objectives. (but strong performance in practice)


# Explain the Chain-rule

The chain-rule is a mechanism used to perform gradient descent (BackPropagation) in a deep neural network. This mechanism is necessary to appropriately attribute error and training to each weight in the network. 

To compute the gradient at the last layer, one does the derivative of the loss with respect to the parameters. This can be decomposed as the derivative of the error (loss) wrt the weights of the network (to each one of them independently).

From: https://en.wikipedia.org/wiki/Backpropagation "Backpropagation computes the gradient of a loss function with respect to the weights of the network for a single input–output example, and does so efficiently, computing the gradient one layer at a time, iterating backward from the last layer to avoid redundant calculations of intermediate terms in the chain rule; this can be derived through dynamic programming."

Generally speaking, the chain rule states that the derivative of two compound functions (the function of a function f(g(x))) is the derivative of the outer function times the derivative of the inner function. f'(g(x))\*g'(x). This idea is very straightforward, and is what we apply to the multiple compounded steps of the net (weights, activation, weights, activation, ...)

There's a very intuitive explanation here: https://en.wikipedia.org/wiki/Chain_rule#Intuitive_explanation

For neural networks it is easy. Each of such functions is a layer. So its change depends on previous layers (the derivative measures change). If layer 3 depends on layer 2, which depends on layer 1

d(L3\*L2\*L1\*X - Y) / d(L1L2L3)

Define derivative (see also Partial derivative): See the formula in here: https://en.wikipedia.org/wiki/Derivative#Notation with u and y, it was the most clear I've seen so far. Define gradient: https://en.wikipedia.org/wiki/Gradient It is the change to f(r) when r changes.

Ok, we definitely want to compute the derivative of the error wrt a specific weight. The idea being that we compute the slopes in the error function, as we wish to traverse it towards lower valleys of the function. In this sense, we want to study how does the error function vary when we vary the weights so we can change the weights to affect the error function in the appropriate direction (the one that interests us, i.e., downwards)

Following bishop's derivation for backpropagation (basically applying the chain rule, see my notes), we start at the output and see that the gradients for the output layer depend on the activations and activateion function of the last layer, and the final output. Then, the gradients of the next to last layer depend on that same thing, and the gradients at output. And so on. Hence, the chain-rule can be used to effectively attribute gradients throguhout the network by computing the partial derivatives from end to start of the network, and reusing the previous layer gradient's at each step so we don't need to compute them again.


# What metrics would you use in a classification problem?
        ◦ A metric to train the network: The most common loss for classification is cross-entropy
        ◦ A metric to measure: Accuracy? Recall? Precision?

So, we've all heard of accuracy. It will tell us how many times the model was correct. But this tells us only limited information about the model.

Acording to this (https://en.wikipedia.org/wiki/Accuracy_and_precision) we can use precission, which, as I understand it, is how sure the model was when making those predictions. So, in a classification problem, precission would be the certainty on that prediction. If argmax gives us the class, max would be the precission on predicting that class. It is not the same if the model was 90% sure it was a dog, and all other classes had like 1%, than if the model was 34% sure it was a dog and 31% sure it was a cat. Do not confuse this definition of precission with the one used in retrieval (where you check precission and recall.) In this sense, Accuracy is the ratio of correctly labeled samples by all samples.

Now, while precission and recall can be used for binary classification (and can be computed per-class in multi-class problems, and latter aggregated as a single statistic of the model). Computing TP, TN, FP, and FN is called a confusion matrix. With these values you can the compute Precission = TP / (TP + FP) (How many of the ones I clasiffied as "positive" are actually positive) and Recall = TP / (TP + FN) (How many of the positives did I actually get).

Finally, we've got the F1-socre, which computes the harmonic mean (??) of precission and recall. F1 = 2\*(P\*R / P+R)

Accuracy (TP + TN / TP + TN + FP + FN) may have undesirable effects as it aggregates too much information. For example, if the dataset is unbalanced, a model that always predicts the majority class will still be mostly accurate. There are alternatives that balance the TP and TN depending on the number of samples per class bAcc = (TPR + TNR)/2... I could keep going, but https://en.wikipedia.org/wiki/Precision_and_recall already has a very complete table of metrics.

Apparently you can choose one metric or another depending on the cost of FP and FN. If one is more costly than the other, some metrics can be better indicative is the model is good or not. 

# How to deal with an imbalanced data set
https://medium.com/game-of-bits/how-to-deal-with-imbalanced-data-in-classification-bd03cfc66066
"decision tree-based algorithms perform well on imbalanced datasets. Similarly bagging and boosting based techniques are good choices for imbalanced classification problems."

You could approach it as an anomaly detection problem, to make

Assuming this is a labeled dataset, you can weight the loss differently for those classes for which you have less data. 

You could also force to train more on those classes for which you have less samples, or, alternatively, sample less the classes for which you have more, trying to achive that the network sees all classes more or less the same number of times. Problems: if you undersample you're effectively loosing data, and if you oversample you may overfit those samples that the net sees multiple times. 

SMOTE: You can extend your minority class with synthetic data (interpolating samples) there are various methods, each with its pros and cons.

And I'm guessing that you'd also need to consider that imbalance when computing the metrics to evaluate the model (e.g., Using recall for the minority class(es))

# What loss function will you use to measure multi-label problems
        ◦ I’ve never seen multi-label problems!

# Explain PCA intuitively. When is it useful?

# What causes vanishing / exploding gradients? How to solve it?

# Ensemble algorithm? (Random forest; feature and data replacement; reduce variance
"explain random forests"

# How to split a tree?

# How does a typical anomaly detection pipeline work?

# Let’s say now you want to identify a threshold for a classifier that predicts whether a customer will sign up to prime or not. What criteria could we use to find the threshold?

# In the model we developed, we have a billion positive samples and 200,000 negative samples. If you were to review our model before we put it on the website, what would you look for in the model to ensure this model is not bad?

# Explain Maximum likelihood.

# Explain Expectation Maximization.

# ReLU? GeLU? TanH? Sigmoid? Softmax? Logistic function (like softmax for binary problems)?

# How does an RNN work? And a GRU unit? How is LSTM different? What limitations do these models have? What strenghts?

# What are the different parameters to control a convolutional layer? 
 - Kernel size, stride, group, dimensions it affects...

# Explain Transposed convolutions.

# What are energy-based models? How are they different from a common loss function?

# In visual generative models, what are good metrics of the results? Explain PSNR, MSE, DSSIM... others?
SE and PSNR are objective measurements of reconstruction quality.
DSSIM is a measure of the perceived quality

# Could you describe how you train this Context-awareness entity ranking model?

# Explain the “Moore–Penrose inverse” and “Frobenius norm”.

# Explain binomial probabilities

# Explain Bayes’ Rule

# What are the most common probability distributions? What are some common scenarios in which you'd use each of them?

# “how would you explain to an engineer how to interpret a p-value?”


CODING QUESTION

# Remove duplicated samples in a dataset that doesn’t fit in memory.
# Find the longest common subsequence.
# “reverse a linked list” or ”invert a binary tree”
# (From Amazon: https://www.amazon.jobs/en-gb/landing_pages/p-software-development-topics) Consider revising common algorithms such as traversals, divide and conquer, breadth-first search vs. depth-first search and make sure you understand the trade-offs for each. Knowing the runtimes, theoretical limitations and basic implementation strategies of different classes of algorithms is more important than memorising the specific details of any given algorithm.

# What are mixed precission, gradient clipping




# Other resources:
- fast.ai’s Practical Deep Learning for Coders 
- theoretical course like Machine Learning by Coursera. 
- Stanford: Machine Learning Systems Design (https://stanford-cs329s.github.io/)
- Deep Learning by Goodfellow et al. (https://www.deeplearningbook.org/)
- Machine Learning: A Probabilistic Perspective by Kevin P. Murphy. (I have it)
- CS231N: Convolutional Neural Networks for Visual Recognition (https://cs231n.stanford.edu/) (especially the parts about gradient descent, activations, and optimizations as well as rewatch Full Stack Deep Learning lectures, especially the ones on Setting up Machine Learning Projects and Infrastructure and Tooling)
- Some more relevant courses (includes some of the above): https://huyenchip.com/ml-interviews-book/contents/4.3.1-courses.html
- Some books and articles (includes some of the above): https://huyenchip.com/ml-interviews-book/contents/4.3.2-books-&-articles.html
- Some other resources for interviews and ML: https://huyenchip.com/ml-interviews-book/contents/4.3.3-other-resources.html
- ML Interviews: https://huyenchip.com/machine-learning-systems-design/toc.html
- MLOps: https://huyenchip.com/mlops/
- LLM Book: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-gpt-to-llama2.ipynb
- Probability cheatsheet (by William Chen): https://static1.squarespace.com/static/54bf3241e4b0f0d81bf7ff36/t/55e9494fe4b011aed10e48e5/1441352015658/probability_cheatsheet.pdf
