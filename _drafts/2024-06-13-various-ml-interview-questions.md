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
 - In this case, memorization could be seen as a shortcut the network is taking. When a network memorizes instead of learning, it leads to a loss of generalization abilities.
 - To check if that is indeed the case, one may have to use K-fold cross-validation. Low error rates and a high variance are good indicators of overfitting.
 - Possible solutions:
	 - Data augmentation: May help to virtually increase the size of the training data and making the model more robust.
	 - Early stopping: Stop before the network has a change to overfit. However, a fine balance has to be found. Stopping too early may not give the network a change to learn the necessary patterns in the data (underfitting).
	 - Adding data: This must be done carefully, not to add even more complexity to the problem! If data is not clean (adds irrelevant information / noise) it may make problems worse by not allowing the network to learn.
	 - Feature selection / Prunning: Detect irelevant features / parameters and remove them, to allow for relevant features to be dected.
	 - Regularization: e.g. Dropout or other methods (lasso regularization, ridge regression)... I've read this reduces noise, but I believe what it does is actually increase it! Forcing the network to become robust to it. Apparently lasso is L1 and ridge is L2 regularization (adding a component to the loss (abs sum or squared sum of weights) that avoids that any particular weight is too high)... "This helps to prevent overfitting by reducing the impact of noisy or irrelevant predictors in the model" but I'm not sure, I feel this would solve the bias problem, no? So, ok, no particular noise signal can dominate, allowing for, at least, a "close-to-uniform" treatment of features, hence when some specific feature is present (a noise one), it won't be able to completely take away the result. But then... what is bias?
	 - Ensemble methods: (e.g. decision trees), train multiple classifiers on different subsets of the data in order to reduce the variance of the problem. Then find a consensus between the multiple models (bagging and boosting). May make sense to also use subsets of features for the different classifiers?

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

Now, while precission and recall can be used for binary classification (and can be computed per-class in multi-class problems, and latter aggregated as a single statistic of the model), it's traditionally something from retreival. Computing TP, TN, FP, and FN is called a confusion matrix. With these values you can the compute Precission = TP / (TP + FP) (How many of the ones I clasiffied as "positive" are actually positive) and Recall = TP / (TP + FN) (How many of the positives did I actually get).

Finally, we've got the F1-socre, which computes the harmonic mean (??) of precission and recall. F1 = 2\*(P\*R / P+R)

Accuracy (TP + TN / TP + TN + FP + FN) may have undesirable effects as it aggregates too much information. For example, if the dataset is unbalanced, a model that always predicts the majority class will still be mostly accurate. There are alternatives that balance the TP and TN depending on the number of samples per class bAcc = (TPR + TNR)/2... I could keep going, but https://en.wikipedia.org/wiki/Precision_and_recall already has a very complete table of metrics.

Apparently you can choose one metric or another depending on the cost of FP and FN. If one is more costly than the other, some metrics can be better indicative is the model is good or not. 

*******
Ok, let's see, I've started to ckech these kind of metrics for Anomaly Detection and I've gone down a heavy spiral. I think any metrics for binary classification can be used in this case, and it's the same for retreival (in the end, could be seen as clasifying a sample as relevant/irrelevant for a given query). In particular, I was trying to understand ROC https://en.wikipedia.org/wiki/Receiver_operating_characteristic, which is a measure for AD. And I found the same huge table as in the Precission/Recall article I linked above.

Let's start by the null-hypothesis (https://en.wikipedia.org/wiki/Null_hypothesis).

The null-hypothesis is the default hypothesis (i.e., the condition is not met, there is not a relationship, etc.). In anomaly detection, the null-hypothesis is that the sample is normal. In retrieval, the sample is not relevant. The sample does not belong to the class. The set of samples that satisfy the null-hypothesis are the "negatives" (meaning that they **do not** satisfy the condition, relationship, etc. stablished by the *alternative* hypothesis). The alternative hypothesis states that the sample satisfies a condition (e.g., to be retrieved, to belong to a given class, or have some anomaly present). A sample that satisfies the alternative hypothesis is hence dubbed a positive sample.

I think all of this is a bit confusing because of the use of null, reject, negations etc.
A couple of examples. In medicine, if you're testing for an illness, a positive result means that, indeed, the condition (presence of the virus) is satisfied. So a positive results indicates that the person is unhealty. Which is a bit counterintuitive if you look at health, but it makes a lot of sense if you look at it from the perspective of the test: to pass the test, the virus must be present. In other words: null-hypothesis, healthy individual, **non**-presence of virus, a sample which satisfies this is a *negative* sample because it does **not** pass the "virus-presence test"; alternative hypothesis, unhealty individual, presence of virus, a sample which satisfies this is a *positive* sample because it does pass the "virus-presence test".

In court, the null-hypothesis is that "someone is inocent (-until proven guilty"). The alternative hypothesis to be proved is that the person is indeed guilty. The test here is "is the person guilty?". If the test is passed (*positive*), the person is guilty (which sounds negative in a laymans view, but again, it's positive in relation to the test), and the null-hypothesis is rejected. If the test fails (*negative*), the person is innocent, meaning that the null-hypothesis is accepted.

In this context we have two types of errors (https://en.wikipedia.org/wiki/Type_I_and_type_II_errors):
 - Type I error: reject null-hypothesis by mistake.
 - Type II error: accept null-hypothesis by mistake.

The key idea for understanding these errors is not from accepting one or the other hypotheses, but looking at it from the null-hypothesis alone (accepted/rejected), and assuming that the alternative one is rejected/accepted by default.

However, if we want to look at it from the context of True/False Positive and True/False negative, we're looking at it from the point of the test, i.e., the alternative hypothesis. So we're looking at it the other way around, which makes it confusing (at least for me).

The set of positive samples is formed by all those in which the test should pass, and really AH is correct (i.e., NH should be rejected, e.g., anomaly, valid retrieved document, correctly identified as belonging to a class...):
  1. (TP) AH correctly accepted --> True Positive (i.e., NH correctly rejected)
  2. (FN) AH incorrectly rejected --> False Negative (i.e., NH incorrectly accepted)

The set of negative samples is formed by all those in which the test fails, and really AH is incorrect (i.e., NH should be accepted, e.g., normal sample, do not retrieve the document, identified as not belonging to a class...):
  1. (TN) AH correctly rejected --> True Negative (i.e., NH correctly accepted)
  2. (FP) AH incorrectly accepted --> False Positive (i.e., NH incorrectly rejected)

Now, whith all this in mind, let's take a look at the most common metrics in this context (look at the chart of metrics I linked above for reference, and many MANY more metrics):

 - **True Positive Rate** (Recall, or Sensitivity): TP / (TP + FN). Ratio between identified positives and all real positives (e.g., identified anomalies divided by all anomalies). It's the conditional probability of passing the test, given that the true condition is positive. Recall: In the context of retrieval, how many of the ones I should bring I am capable of actually retrieving/recalling. Sensitivity: How sensitive is my test, if it's 100% sensitive it means I do not leave any samples which satisfy the condition out. Meaning that if the sample fails the test, I can confidently reject it. A test with a higher sensitivity has a lower type II error rate. 
 - **True Negative Rate** (Specificity): TN / (TN + FP). Ratio between identified negatives and all real negatives (e.g, identified as not anomalies (i.e., identified normal samples) divided by all normal samples). It's the conditional probability of failing the test, given that the true condition is negative. Specificity: How specific my test is, if it's 100%, it means all samples that do not satisfy the condition will be correctly rejected. Meaning that if the sample passes the test, I can confidently accept it. A test with a higher specificity has a lower type I error rate. 
 - **False Negative Rate** (Miss rate, or Type II): FN / (FN + TP). Ratio between the ones that falsely fail and all which pass the test. It's the conditional probability of failing the test, given that the true condition is positive (i.e., probability of falsely rejecting the alternative hypotesis or, in other words, accepting the null-hypothesis).
 - **False Positive Rate** (False alarm, fall-out, or Type I): FP / (FP + TN). Ratio between the ones that falsely pass the test and all which fail the test. It's the conditional probability of passing the test, given that the true condition is negative (i.e, probability of falsely accepting the alternative hypothesis or, in other words, rejecting the null-hypothesis).
 - **Accuracy**:
 - **Precission**:

Now for the cherry on top. In the context of anomaly detection (and I'm sure, other places), a metric noted as Receiver Operatinc Characteristic (https://en.wikipedia.org/wiki/Receiver_operating_characteristic) is used. Instead of measuring a single value, it is based on a set of thresholds. In order to consider a specific result Positive or Negative, one must use a threshold (e.g., a network outputs the probability that a sample is anomalous (in my current context), or that it's relevant for a query in the context of retrieval). 

From the wikipedia article:
 ```
 To draw a ROC curve, only the true positive rate (TPR) and false positive rate (FPR) are needed (as functions of some classifier parameter). The TPR defines how many correct positive results occur among all positive samples available during the test. FPR, on the other hand, defines how many incorrect positive results occur among all negative samples available during the test. A ROC space is defined by FPR and TPR as x and y axes, respectively, which depicts relative trade-offs between true positive (benefits) and false positive (costs).
 ```

By sliding the threshold one should be able to achieve all possible values for both TPR and FPR. The relation between the two by the given model will draw a specific curve. In general, the higher the area under the curve, the better the model.

See also:
 - https://en.wikipedia.org/wiki/Sensitivity_and_specificity
 - https://en.wikipedia.org/wiki/Precision_and_recall
 - https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Fall-out
 - https://en.wikipedia.org/wiki/False_positives_and_false_negatives#false_negative_rate
 - https://en.wikipedia.org/wiki/False_positive_rate

# How to deal with an imbalanced data set
https://medium.com/game-of-bits/how-to-deal-with-imbalanced-data-in-classification-bd03cfc66066
"decision tree-based algorithms perform well on imbalanced datasets. Similarly bagging and boosting based techniques are good choices for imbalanced classification problems."

You could approach it as an anomaly detection problem, to make the model focus on those anomaly cases.

Assuming this is a labeled dataset, you can weight the loss differently for those classes for which you have less data. 

You could also force to train more on those classes for which you have less samples, or, alternatively, sample less the classes for which you have more, trying to achive that the network sees all classes more or less the same number of times. Problems: if you undersample you're effectively loosing data, and if you oversample you may overfit those samples that the net sees multiple times. 

SMOTE: You can extend your minority class with synthetic data (interpolating samples) there are various methods, each with its pros and cons.

And I'm guessing that you'd also need to consider that imbalance when computing the metrics to evaluate the model (e.g., Using recall for the minority class(es))

# What loss function will you use to measure multi-label problems

## About multi-label problems
https://en.wikipedia.org/wiki/Multi-label_classification#Statistics_and_evaluation_metrics
https://medium.com/data-science-in-your-pocket/multi-label-classification-for-beginners-with-codes-6b098cc76f99
Multi-label classification problems are generally posed as either an esemble of binary classifiers (one vs all for all classes), or, alternatively, one can build pseudo-classes for all possible combination of labels and then train a common multi-class model. These problems may overcomplicate issues like class imbalance (if we need more examples of a specific label, we may not be able to upsample the minority without also upsample for other associated labels -- and the same if we need to downsample the majority class, we may be distroying relevant samples for other classes.). Apparently, you could also simply use a normal classifier and replace the output softmax by a sigmoid, so multiple entries may have a large activation. I guess in that case you'd need to define a threshold from which you'd consider that the model has detected one of the classes. I think that, as with other cases, the label won't be a 1-hot vector but a binary vector. (Apparently not, you need to use a binary cross-entropy, aka logistic loss, (and sum it over the multiple classes) for some reason, acording to [this](https://machinelearningmastery.com/multi-label-classification-with-deep-learning/). May have somehting to do with the emphasis on having a sigmoid for each output neuron -- one for each class). https://medium.com/@kitkat73275/multi-label-classification-8d8ae55e8373


## Metrics
Now, if we uses the pseudo-class variant, we can straight use accuracy, meaning that we'll only count as valid if the model correctly predicts all labels for a given sample. We can also apply this if we use multiple binary classifiers. Now, this is very strict, and we may want to value when our model predicts at least some of the labels. For this we can use:

 - Hamming distance (loss): simply counts how many correct labels for each sample and normalizes by the batch size and number of max labels for a given sample in the batch. We represent each label as a binary vector, and we count the hits by applying an *xor* gate: "returns zero when the target and prediction are identical and one otherwise".
 - Precission, recall and F1: Here it is easier to use these metrics similarly to a retrieval problem (one has to *retrieve* all labels for a given sample). In this sense, T are the true labels, and P are the predicted ones. Precission: intersec(T,P) / P (i.e., true positives divided by true positives + false positives); Recall: intersec(T,P) / T (i.e., true positives divided by true positives + false negatives). F1 is the harmonic mean of the two.
 - Jaccard Index (IoU): intersec(T, P) / union(T, P) (true positives divided by true positives + false negatives + false positives)
   IoU is better understood in the context of image segmentation. It computes the ratio of success (intersection of prediction and ground truth)with respect to the overall area (predicted + ground truth)


# Ensemble algorithm? (Random forest; feature and data replacement; reduce variance
https://en.wikipedia.org/wiki/Ensemble_learning
"Fundamentally, an ensemble learning model trains many (at least 2) high-bias (weak) and high-variance (diverse) models to be combined into a stronger and better performing model. Essentially, it's a set of algorithmic models — which would not produce satisfactory predictive results individually — that gets combined or averaged over all base models to produce a single high performing, accurate and low-variance model to fit the task as required."

"Ensemble learning typically refers to Bagging (bootstrap-aggregating), Boosting or Stacking/Blending techniques to induce high variability among the base models. Bagging creates diversity by generating random samples from the training observations and fitting the same model to each different sample — also known as "homogeneous parallel ensembles". Boosting follows an iterative process by sequentially training each next base model on the up-weighted errors of the previous base model's errors, producing an additive model to reduce the final model errors — also known as "sequential ensemble learning". Stacking or Blending consists of different base models, each trained independently (i.e. diverse/high variability) to be combined into the ensemble model — producing a "heterogeneous parallel ensemble"."

It seems like the key here is to have variability in your models, to make sure you're covering a wider range of solutions: "although perhaps non-intuitive, more random algorithms (like random decision trees) can be used to produce a stronger ensemble than very deliberate algorithms"

It may take many forms: Bagging (Random forests), Boosting, different Bayesian approximation techniques (B optimal classifier, B averaging, B model combination...), Bucket of models...

Bagging: Create various sub-datasets, allowing to have the same sample multiple times, others may not appear. With each of these you can train a random tree to make decisions based on that set of the data. By building several noisy (variable) models, with different biases, we get to cover varying possibilities. The final decission is reached by voting (for example).
See also: Rotation forest – in which every decision tree is trained by first applying principal component analysis (PCA) on a random subset of the input features

Boosting: Build a sequential set of classifiers where each focuses on tackling the samples for which the previous one had more trouble (more errors.) The second one will see all samples the same, but the ones for which the first model had more trouble will be weighted higher in the loss. This is done multiple times. Tend to overfit (example is Adaboost).

# What is a decision tree?
https://en.wikipedia.org/wiki/Decision_tree
A decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes). The paths from root to leaf represent classification rules. 

https://en.wikipedia.org/wiki/Decision_tree_learning
"process of top-down induction of decision trees (TDIDT)[5] is an example of a greedy algorithm, and it is by far the most common strategy for learning decision trees from data"

# Explain RandomForests
https://en.wikipedia.org/wiki/Random_forest
"because it is invariant under scaling and various other transformations of feature values, is robust to inclusion of irrelevant features, and produces inspectable models. However, they are seldom accurate"

"Trees that are grown very deep tend to learn highly irregular patterns: they overfit their training sets, i.e. have low bias, but very high variance. Random forests are a way of averaging multiple deep decision trees, trained on different parts of the same training set, with the goal of reducing the variance. This comes at the expense of a small increase in the bias and some loss of interpretability, but generally greatly boosts the performance in the final model."

Ensembling random trees aids with these problems.


# How to split a tree?
https://www.analyticsvidhya.com/blog/2020/06/4-ways-split-decision-tree/
"The process of recursive node splitting into subsets created by each sub-tree can cause overfitting. Therefore, node splitting is a key concept that everyone should know."

"The ways of splitting a node can be broadly divided into two categories based on the type of target variable:

    1. Continuous Target Variable: Reduction in Variance
    2. Categorical Target Variable: Gini Impurity, Information Gain, and Chi-Square"


https://www.geeksforgeeks.org/how-to-determine-the-best-split-in-decision-tree/
    1. Calculate Impurity Measure:
        Compute an impurity measure (e.g., Gini impurity or entropy) for each potential split based on the target variable’s values in the resulting subsets.
    2. Calculate Information Gain:
        For each split, calculate the information gain, which is the reduction in impurity achieved by splitting the data.
    3. Select Split with Maximum Information Gain:
        Choose the split that maximizes information gain. This split effectively separates the data into subsets that are more homogeneous with respect to the target variable.
    4. Repeat for Each Attribute:
        Repeat the process for all available attributes, selecting the split with the highest information gain across attributes.


hacer algún ejercicio de programación al respecto

# What is Least Squares? What is Least Sequares error? I have a linear system x = wy. How would you fit it other than gradient descent?
Least squares is a minimization technique for linear models it is used by minimizing the sum-of-squares error, most suitable for regression problems (not classification, duh). (It can be used for non-linear models, but... "The linear least-squares problem occurs in statistical regression analysis; it has a closed-form solution. The nonlinear problem is usually solved by iterative refinement; at each iteration the system is approximated by a linear one, and thus the core calculation is similar in both cases." https://en.wikipedia.org/wiki/Least_squares#Non-linear_least_squares)

I think I'm a bit confused. 
So here it seems that a linear model is just a function that maps any new point onto a line?
No, we're trying to find the linear function (i.e., a line) that best describes the data, i.e., which minimizes the compounded distance to all data points. SO! How does that help us when we want to build a classifyer? 
A linear model I thought what did was draw a line, and all points on one side belong to a specific class. Ah! But we're talking about regression now!! Of course! We can now give a data point in x axis and get a response from our line in the y axis.

SEGUIR LEYENDO BISHOP PG. 105

For a geometric interpretation of what's going on, Bishop's book (around p. 103) and this post (https://math.stackexchange.com/questions/1298261/difference-between-orthogonal-projection-and-least-squares-solution)
Let's see, it seems like for linear equations, you can, instead of directly minimizing Ax = b --> x = b - A, you can just work with the projection of b onto A.
A least squares solution of the system Ax = b is a vector x such that Ax is the orthogonal projection of b onto the column space of A. It is not the orthogonal projection itself.
Básicamente, la "sombra" de b sobre el espacio que queda definido por las columnas de A, es igual a Ax.

From "Bayesian Reasoning and ML" book, pg 332
In regression we minimize the residuals – the vertical distances from datapoints to the line. In PCA the fit minimizes the orthogonal
projections to the line. That's why sometimes PCA is called orthogonal regression. So... To compute the Moore-Penrose you have to perform SVD... so probably there is a strong relationship between PCA (for which you also compute SVD).

The line wx - b

It is called the least squares solution because you cannot guarantee an exact solution, but the closest one? So the solution that has the least squared distance to the actual solution?

What we really want to do is to fit a line that best fits the samples we've got. As this is a linear problem, we can see that the output of the model/function/network is nothing else than a linear combination of the inputs.

We want to find a function such that its parameters map 


To actually compute a solution we will need to compromise, as the actual solution lies outsied of the plane defined by the input elements, which is the only placew here the output of our model (solution to our problem) we need to use the least squares. Apparently, least squares is a way to find a solution to a set of linear equations

[Thomas Schmelzer said](https://www.linkedin.com/pulse/note-normal-equations-thomas-schmelzer/):
> There's an exact solution to this problem. The normal equations are A^T A x = A^T b where ^T denotes the transpose. NG's celebrated machine learning course and Geron's book both claim that trying to solve x = inv(A^T A) \* A^T b is a bad idea and one should rather work with gradient descent on the Euclidean norm of the residual r = A\*x - b. They argue that computing the inverse is expensive (slow) and it will consume tons of memory.
>
> First, let's rewrite the normal equations as A^T (A\*x - b) = 0. This equation is far closer to the name. It says that the residual is orthogonal on all columns of A and hence the residual is orthogonal (or normal) to the image of A. In particular, there is no inverse in this equation.


Another view of the geometric interpretation: https://fncbook.github.io/fnc/leastsq/normaleqns.html
>The vector in the range (column space) of 𝐀 that lies closest to 𝐛 makes the vector difference 𝐀𝐱−𝐛 (i.e., the residual) perpendicular to the range. Thus for any 𝐳, we must have (𝐀𝐳)𝑇(𝐀𝐱−𝐛)=0, which is satisfied if 𝐀𝑇(𝐀𝐱−𝐛)=𝟎.

The inverse of a matrix is [the "reciprocal"](https://www.mathsisfun.com/algebra/matrix-inverse.html). In scalar maths, we divide, but we cannot do such a thing as dividing, so we "multiply by the inverse". Now, not all matrices have an inverse. For a matrix to be invertible it must be square and non-singular (have a determinant >0). With A having an inverse B, the key idea is that AB = BA = I. Some rectangular matrices can be left / right invertible, depending on some properties of the rank, but not fully invertible (either AB = I or BA = I, but AB != BA). ([Source](https://en.wikipedia.org/wiki/Invertible_matrix#Definition). When talking about pseudoinverses is because they check some of the requirements of inverses but not all. [Clive Hunt said](https://www.quora.com/What-is-the-difference-between-the-pseudo-inverse-and-inverse-of-a-matrix)):
>There is no “difference” between them in the sense that, if a matrix has an inverse, then it will also have a unique pseudo-inverse that will be the same as the inverse. As long as they both exist, they have to be the same.
>
>But many matrices do not have an inverse: non-square matrices, and square matrices with zero determinant (“non-invertible”). All of these, even including a rectangular matrix full of zeroes, nevertheless have a psuedo-inverse.

So, when trying to solve least squares, an equation system, if we need to isolate the variable to solve the system, we may need to send matrices to the other side. Why use the pseudoinvers instead of the inverse? I guess we cannot guarantee that A (the space formed by our input samples) has an inverse. For starters, it will probably be rectangular (different number of samples than features). So it is safer to asume that it is not, and work with the pseudoinverse. ([See here](https://math.stackexchange.com/questions/435208/moore-penrose-inverse-and-standard-inverse)). Finally, the Moore-Penrose pseudoinverse still holds some of the properties that are necessary for this particular case, and indeed, one of the main uses for this technique is actually solving least squares.




More links:
https://mathworld.wolfram.com/NormalEquation.html

# Differences / Problems / Advantages of L1 and L2?

# Explain different generative models. In which situations would you use each? Pros/Cons of each.

AE vs VAE, vs GAN, vs Diffusion.

# Can you use noise different than gaussian for diffusion models? Why would you want to do that?

# Explain Expectation Maximization. Whas is it good for? When does it fail?

# Explain Maximum likelihood. Whas is it good for? When does it fail? How is it related to the cross entropy? 

# Explain PCA intuitively. When is it useful? When does it fail?
programar PCA

# Structure from motion? NeRFs? Radiance fields? Gaussian splatting? 

######################## Hasta aquí cositas de Amazon

# What are SVM? 

# Explain A\*. What other planning methods exist? Pros, cons etc. 

# Do you really understand entropy? Information theory

# What causes vanishing / exploding gradients? How to solve it?
entender la idea básica
hacer ejercicios básicos de pytorch de wrappear lo que toque para lograr eso

# How does a typical anomaly detection pipeline work?
https://github.com/bitzhangcy/Deep-Learning-Based-Anomaly-Detection

# Let’s say now you want to identify a threshold for a classifier that predicts whether a customer will sign up to prime or not. What criteria could we use to find the threshold?

# In the model we developed, we have a billion positive samples and 200,000 negative samples. If you were to review our model before we put it on the website, what would you look for in the model to ensure this model is not bad?

# What is a Kolmorogov-arnold network? How is it different from a traditional one? 

# Explain Logistic Regressions

# ReLU? GeLU? TanH? Sigmoid? Softmax? Logistic function (like softmax for binary problems)?

Hacer lo mismo que con la softmax (código de visualizar) para entender un poquito las diferentes cositas que hay

I think the logistic function is the binary cross-entropy.

# How does an RNN work? And a GRU unit? How is LSTM different? What limitations do these models have? What strenghts?

# What are the different parameters to control a convolutional layer? 
 - Kernel size, stride, group, dimensions it affects...

# Explain Transposed convolutions.

# What are atrous convolutions?

# Explain convolutions, how do they work? how are they applied? What properties do they have that make them different to a FC network?

# Explain some recent advancements to CNNs.

# What are some widespread CNN architectures? What are the advantages of each of them?

# What are energy-based models? How are they different from a common loss function?

# What is optical flow?

# Explain Kalman filters.

# In visual generative models, what are good metrics of the results? Explain PSNR, MSE, DSSIM... others? Frechet Inception Distance?
MSE and PSNR are objective measurements of reconstruction quality.
DSSIM is a measure of the perceived quality

# Could you describe how you train this Context-awareness entity ranking model?

# Explain the “Moore–Penrose inverse” and “Frobenius norm”.
https://mathworld.wolfram.com/FrobeniusNorm.html
The Frobenius norm is the Euclidean norm but for matrices. It's the square root of the sum of squares of its components.

# Explain binomial probabilities

# Explain Bayes’ Rule

# What are the most common probability distributions? What are some common scenarios in which you'd use each of them?

# “how would you explain to an engineer how to interpret a p-value?”

# What is Layer Norm and how does it work? How is it different to BatchNorm? Which other normalizations are out there?

https://user-images.githubusercontent.com/76557164/176494819-d7be7cd7-a181-417b-aeee-b197c86a2a18.png

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
