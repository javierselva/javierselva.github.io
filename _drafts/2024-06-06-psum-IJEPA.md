---
layout: post
section-type: post
has-comments: false
title: "I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"
category: paper-summary
tags: ["tutorial"]
---
# Introduction
When LeCun first published his vision of how an embodied agent should function in the world I was excited. It is always nice to escape briefly into reading a little bit of theories and hypothesis on how things should work. It is important, I believe, to take a step back and make sure we're still going in the direction we wish to follow, instead of banging our heads blindly against the next engeneering problem.

In his 60 page monograph titled "[A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)", LeCun introduces his ideas on what would be required to build an embodied agent into our world. One of the key cornerstones of this proposal are *Joint Embedding Predictive Architectures* (JEPA). The key idea here is the use of siamese networks (a couple of networks sharing architecture and potentially the weights) so that one receives an input and the other receives a slightly different input (either another part of the same input or a slightly modified version of it). The network is then trained so that the outputs from both of them should be predictible from one another (I recommend reading [this blog post to get the general idea](https://ai.meta.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/) if you don't feel like reading a 60 page monograph). 

INCLUDE IMAGE HERE

This is far from being something entirely new. Far from it, what LeCun was proposing was a generalization of many different self-supervised learning mechanisms. From [SimSiam (PDF)](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.pdf), to [BYOL (PDF)](https://papers.nips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf), many self-supervised objectives for vision can be cast under this umbrella. Recently, LeCun himself has worked on several papers that implicitly try to be instantiations of the JEPA idea, such as VICReg or BarlowTwins.

Maybe an intro to energy based?? It would be helpful for me to fully understand

# I-JEPA

I-JEPA has been the first work to explicitly instantiate a JEPA architecture for image self-supervised training.

Key ideas:
	- No need to use data augmentation (for video this is cool, as they are very expensive). IMO, you still need the views, which in many cases is going to still be a problem, depending on how sensitive is your model to them.
	- Instead it does the local-global thing!! I already have this explained in the survey, maybe I can have some quotations
		- Demystifying Contrastive Self-Supervised Learning: Invariances, Augmentations and Dataset Biases
		- Video Representation Learning by Dense Predictive Coding
		- BraVe: Broaden Your Views for Self-Supervised Video Learning
		- Self-Supervised Video Transformer 
		- Efficient Self-supervised Vision Transformers for Representation Learning
		- Learning Representations by Maximizing Mutual Information Across Views
		- 
		- They even comment on this in the abstract: "masking strategy; specifically, it is crucial to (a) sample target blocks with sufficiently large scale (semantic), and to (b) use a sufficiently informative (spatially distributed) context block"
		- Nice representations for high semantic level, but seem to fail in specific cases (low-level reasoning or unbalanced data): low-level may require different invariances.

	- The key idea is to predict specific regions given some context. This is very similar to other masking strategies. You've got your MAE etc. However, instead of trying to reconstruct the input, hog features or using a contrastive objective, here the predictions are made in the cheaper feature space.
	- They show better performance than trying to reconstruct pixels, and seem to be better for low level tasks that other similar methods.

	- Minimize information redundancy across embeddings (VICReg, BarlowTwins) vs maximize entropy of average embedding (MSN, DINO... clustering methods). Importantly, both try to maximize invariance to certain data augmentations. The difference is how they avoid collapse. The former wants to enforce that not two features are correlated while the latter force the samples (actually, the cluster centers), to be uniformly distributed (do they?), effectively achieving the same thing. If two of the features were correlated, there would be parts of the space where no sample would be, hence reducing the objective of a uniform distribution.

	Check!! DINO Says " As shown experimentally in Appendix, centering prevents one dimension to dominate but encourages collapse to the uniform distribution, while the sharpening has the opposite effect". In the MSN paper they claim they are "maximizing entropy", whereas DINO does centering and sharpening: centering makes representations uniform (by subtracting the running mean, salient features will progressively get flattened) whereas sharpening makes them focus on salient features alone (a very low temperature on the softmax will practially put everything to 0 except a few salient features). the balance between the two achieves representations that are different from each other, achieving, I guess, something similar as maximizing entropy: no patterns in representation structure

The method:
	- To avoid collapse they use asymetries and condition the prediction on the positions that are requested to be predicted (by using as input the PE learned for that position)







# Future

LeCun has other papers on this topic such as the VICRegL.

Maybe complement this post with the V-JEPA which is a straightforward adaptation to this one, and comment on the limitations for video.


# TRYING TO UNDERSTAND TEMPERATURE IN SOFTMAX ONCE AND FOR ALL

Softmax function is trying to highlight the higher values. Enter temperature. If you use no temperature, that's equivalent to using a temperature of 1, meaning the distribution is unchanged. If you use a decimal temperature (smaller than one), the lower it is, this is equivalent to multiplying for increasing numbers. If you use an integer, that means you are dividing. When multiplying, bigger values in the vector will proportionally scale more than smaller values, causing the differences between values to become steeper, further highlighting the bigger ones. On the other hand, if using integers, the larger they are, the result of the softmax progresively becomes a uniform distribution.

So, let's see the formula for the softmax

$$ \mathcal{S}(y_i)=\frac{e^{y_i}}{\sum_{j=1}^{i}{e^{y_j}}} $$


```python
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
```

ALSO INSERT GRAPHIC SHOWING THE SOFTMAX and exponential function e

https://www.researchgate.net/profile/Shixiang-Gu-3/publication/309663606/figure/fig4/AS:650391326834690@1532076784734/The-Gumbel-Softmax-distribution-interpolates-between-discrete-one-hot-encoded-categorical.png

https://upload.wikimedia.org/wikipedia/commons/c/c6/Exp.svg



This helps explain how this value can influence the "creativity" of a chatbot such as GPT. The lower the temperature, the less options it has to choose from when producing the next token, hence become more deterministic and *focused*, whereas with a higher temperature it becomes more "diverse" and *creative*, because it weights more options as having a similar weight.