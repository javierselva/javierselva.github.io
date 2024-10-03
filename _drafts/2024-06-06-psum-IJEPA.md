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
	- To avoid collapse they use asymetries and condition the prediction on the positions that are requested to be predicted (by using as input the PE learned for that position).
	- "JEPAs do not seek representations invariant to a set of hand-crafted data augmentations, but instead seek representations that are predictive of each other when conditioned on additional information z"
	- They use two networks, the target encoder and the context encoder (plus a predictor to map representations between the two). From a given input image divided into patches, some of these are selected to be the target. From the remaining bits of the image, multiple blocks are selected to be used as context. None of the context blocks overlap with the target block, in order to avoid trivial solutions where the network simply forward input information to the output. 
	- The whole image is fed through the targegt network to produce representations for all patches in the input. The target patches are then selected from the highly semantic output representations. The different context blocks are then independently run through the context network to produce context representations.
	- The representations of the different contexts are used to predict the target. The context and prediction network are trained on-line through backpropagation, whereas the target network is an exponential moving average of the context one. The training objective is the L2 loss between the predicted patches and those same patches from the original representation.

	- In my opinion, while they do drop data agumentation to build invariant representations, they still require multi-view (and hence, multiple passes through the network). Still way cheaper, specially for video, to crop and cut than having to modify or apply complex pixel-wise functions to alter the content of the image/video. And I mention this because they explicitly say "Common to these approaches is the need to process multiple usergenerated views of each input image, thereby hindering scalability. By contrast, I-JEPA only requires processing a single view of each image." And in my opinion, cropping is an operation through which you produce views. I believe that the thing here is they reverted to multi-crop as the only method to produce views and that still seems to work reasonably well with a few views, different from contrastive approaches that have consistenly shown to necessitate large negative sets. [REF] I think one key point for this to work is that they are not trying to make representations invariant to perturbations, but to make the representations of partial views of an input predictable from each other, making them context-dependent.

	- The networks are ViT-like architectures. This allows for variable number of patches in different runs through the network, meaning that differently-sized context and/or targets can be used. This is also important because they do concatenate learned masked tokens enhanced with positional information to the context tokens, in order to signal the predictor network which patches are to be predicted.

RESULTS
# TOCHECK how many weights do the different vits have??
 - They seem to provide with competitive enough results. I have one problem tough, they make a lot of claims that I-JEPA is better than comparable methods here and there, but in many such occasions that is by means of a larger network, input resolution or smaller patch resolution (which has been shown to help up to a degree [ref]). They argue that, being lighter to train (less augmentation etc.) they can afford to go the extra mile in the other direction and make the models bigger, but I am still sceptic unless I see some FLOP stats which they do not report. What is true, however, is that it seems like JEPA does learn faster (i.e., in less epochs), meaning that a larger model, that in general will require more epochs is in fact feasibly trainable thanks to JEPA... It would seem like invariance learning takes a lot of time. NO!! That is not true!! IJepa does learn faster that other non-augmentation models, but is comparable with data-augmentation-based models.

 - I guess DA is bad because: it is used to learn invariances instead of patterns in the data (?) I guess... but also because it is expensive, no?

 - They perform multiple experiments:
 	- Linear evaluation on Image-Net-1K: I-JEPA seems to outperform most non-data-augmentation methods, except CAE (ref), for which it shows competitive results. When comparing to methods that do use data-augmentation, to me, it is good enough that you can achieve competitive performance with less compute and requiring just a few crop-based views, but in order to outperform data augmentation they need a huge model with higher resolution (or smaller patch size).
 	"Compared to popular methods such as Masked Autoencoders (MAE) [35], Context Autoencoders (CAE) [21], and data2vec [7], which also do not rely on extensive hand-crafted data-augmentations during pretraining, we see that I-JEPA significantly improves linear probing performance, while using less computational effort (see section 7). By leveraging the improved efficiency of I-JEPA, we can train larger models that outperform the best CAE model while using a fraction of the compute. I-JEPA also benefits from scale; in particular, a ViT-H/16 trained at resolution 448 x 448 pixels matches the performance of viewinvariant approaches such as iBOT [75], despite avoiding the use of hand-crafted data-augmentations."
 	- Linear evaluation on Image-Net-1K-1% (only 1% of labeled samples, resulting in around 13 samples per class): These results seem more interesting to me, as when using 12-13 examples per class during the supervised stage, it does seem that this pre-training method does better than the other ones. Still I would rather use the word "competitive" if we look at the L model, it is competitive with data2vec in the sense it requires less epochs (1000 less, which is no small feat), at the expense of 4% accuracy drop; all while outperforming MAE with equivalent model size and still less training). Going to the huge model with smaller patches (more expensive model, makes it tie with data2vec with regards to accuracy, while further reducing the training needs. However the resulting model is going to be more expensive due to the extra flops required. I'd say I rather have a light model that takes more to train (which I am going to do once) than a very expensive model that's cheaper to train. Don't take me wrong, both things are desirable, but if that is the tradeoff when comparing two specific models, I'd say the other wins... But see my next point. Finally, I think it is important to note how the only way to defeat the data augmentation works is through increasing the resolution of the huge model. In this case it is also competitive with regards to the number of epochs, but the fact that these data augmentations are probably using more views (If I recall correctly DINO and friends used around 32 views or such) they are much much more expensive to train. If we look to comparable models, I-JEPA is not really much better (difficult to compare, ViT-L vs ViT-B, but is providing with a method that does not require a super heavy training regime (helping democratize the training of such systems)
 	- Transfer learning to CIFAR100 and Places205: While they still use a huge ViT while some (most) others don't, the differences here are substantial to not possibly be attributed only to model size. This makes the model outperform other non-augmentation dependent methods and very competitive with data-augmentation methods. This generalization abilities also talk good about I-JEPA over other methods. Would be nice to see what they do to try and derive some insights.
 	- Linear probing to object counting and depth prediction: These results make quite a lot of sense. The best performing one (MAE) for these tasks is the one that actually goes full low-level during pre-training and produce a model capable of generating pixels. Followed by I-JEPA, which is trying to preserve low-level information by working at patch level. And the ones that do view invariance, and which are learning high level features perform worse, but not that much worse, mind you... Well... DINO tried to balance local-global by using crops, which could explain why it is better than iBOT even when using a smaller model... Take with a grain of salt, because they are well performing when you consider they are using the base/large architectures where the two best perfroming in this table are using the bigger-more expensive model.

FALTA POR LEER/RESUMIR SECCIONES 7-10



# Future

LeCun has other papers on this topic such as the VICRegL.

Maybe complement this post with the V-JEPA which is a straightforward adaptation to this one, and comment on the limitations for video.


