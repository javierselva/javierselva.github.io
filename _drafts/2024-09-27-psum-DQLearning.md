---
layout: post
section-type: post
has-comments: false
title: "DeepQlearning"
category: paper-summary
tags: ["paper-summary"]
---
# Introduction
When LeCun first published his vision of how an embodied agent should function in the world I was excited. It is always nice to escape briefly into reading a little bit of theories and hypothesis on how things should work. It is important, I believe, to take a step back and make sure we're still going in the direction we wish to follow, instead of banging our heads blindly against the next engeneering problem.

In his 60 page monograph titled "[A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)", LeCun introduces his ideas on what would be required to build an embodied agent into our world. One of the key cornerstones of this proposal are *Joint Embedding Predictive Architectures* (JEPA). The key idea here is the use of siamese networks (a couple of networks sharing architecture and potentially the weights) so that one receives an input and the other receives a slightly different input (either another part of the same input or a slightly modified version of it). The network is then trained so that the outputs from both of them should be predictible from one another (I recommend reading [this blog post to get the general idea](https://ai.meta.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/) if you don't feel like reading a 60 page monograph). 

<figure>
    <img src="/assets/img/blog/paper_summaries/ijepa0.png"
         alt="AltCaption from original paper: A diagram explaining the JEPA architecture. The Joint-Embedding Predictive Architecture (JEPA) consists of two encoding branches. The first branch computes sx, a representation of x and the second branch sy a representation of y. The encoders do not need to be identical. A predictor module predicts sy from sx with the possible help of a latent variable z. The energy is the prediction error." style="border-width: 100px; border-color: white;">
    <figcaption><center><em>Joint-Embedding Predictive Architecture (JEPA). <a href="https://openreview.net/pdf?id=BZ5a1r-kVsf">Source</a>. </em></center></figcaption>
</figure>

This is far from being something entirely new. Far from it, what LeCun was proposing was a generalization of many different self-supervised learning mechanisms. From [SimSiam (PDF)](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.pdf), to [BYOL (PDF)](https://papers.nips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf), many self-supervised objectives for vision can be cast under this umbrella. Recently, LeCun himself has worked on several papers that implicitly try to be instantiations of the JEPA idea, such as [VICReg](https://openreview.net/forum?id=xm6YD62D1Ub) or [BarlowTwins (PDF)](http://proceedings.mlr.press/v139/zbontar21a/zbontar21a.pdf).

Last year we finally go to see the first implementation of this theoretical idea in the form of I-JEPA.

<!-- Maybe an intro to energy based?? It would be helpful for me to fully understand -->

# I-JEPA

[I-JEPA](https://arxiv.org/abs/2301.08243) has been the first work to explicitly instantiate a JEPA architecture for image self-supervised training. The idea is clear: we want to reconstruct missing parts of the input to force the network to learn relevant patterns in the data in order to solve the task. Now, this is different from the traditional inpainting because of the granularity that transformers provide. As short self-quote from "[Video Transformers: A survey.](https://arxiv.org/abs/2201.05991)":

> MTM [(Masked Token Modeling)] could be seen from the lens of generative-based pre-training as it bears great resemblance with CNN-based inpainting. We believe that the success of MTM may be attributable to Transformers providing explicit granularity through tokenization. In order to *conquer* the complex global task of inpainting large missing areas of the input, MTM *divides* it into smaller local predictions. [...] Intuitively, the model needs an understanding of both global appearance [...] as well as low-level local patterns to properly gather the necessary context to solve token-wise predictions. This may allow VTs to learn more holistic representations (i.e. better learning of part-whole relationships).

Also, although it had already been done for Transformers (e.g., MAE or SimMIM), the key novelty with regards to that is that here the predictions are done in feature space (instead of pixel space or HOG features, respectively) by leveraging a siamese network that will produce complete input representations.

<figure>
    <img src="/assets/img/blog/paper_summaries/ijepa1.png"
         alt="AltCaption from original paper: The Image-based Joint-Embedding Predictive Architecture uses a single context block to predict the representations of various target blocks originating from the same image. The context encoder is a Vision Transformer (ViT), which only processes the visible context patches. The predictor is a narrow ViT that takes the context encoder output and, conditioned on positional tokens (shown in color), predicts the representations of a target block at a specific location. The target representations correspond to the outputs of the target-encoder, the weights of which are updated at each iteration via an exponential moving average of the context encoder weights." style="border-width: 100px; border-color: white;">
    <figcaption><center><em>Image-based Joint-Embedding Predictive Architecture (I-JEPA). <a href="https://arxiv.org/abs/2301.08243">Source</a>. </em></center></figcaption>
</figure>


<!--
- Nice representations for high semantic level, but seem to fail in specific cases (low-level reasoning or unbalanced data): low-level may require different invariances.
	- They show better performance than trying to reconstruct pixels, and seem to be better for low level tasks that other similar methods.

	- Some references that helped me form this insight:
		- Demystifying Contrastive Self-Supervised Learning: Invariances, Augmentations and Dataset Biases
		- Video Representation Learning by Dense Predictive Coding
		- BraVe: Broaden Your Views for Self-Supervised Video Learning
		- Self-Supervised Video Transformer 
		- Efficient Self-supervised Vision Transformers for Representation Learning
		- Learning Representations by Maximizing Mutual Information Across Views
		- 
-->
## Core concept
They use two networks, the target encoder and the context encoder (plus a predictor to map representations between the two). From a given input image divided into patches, some of these are selected to be the target. From the remaining bits of the image, multiple blocks are selected to be used as context. The network is tasked with solving a predictive task: from each context block predict the target block.

<figure>
    <img src="/assets/img/blog/paper_summaries/ijepa2.png"
         alt="Table displaying original images and different possible crops to be used either as target or context." style="border-width: 100px; border-color: white;">
    <figcaption><center><em>Examples for the crops used for target and context. <a href="https://arxiv.org/abs/2301.08243">Source</a>. </em></center></figcaption>
</figure>


## Key ideas
- Split the image into tokens. Feed all of them to a *target network* and produce a contextualized representation for each token. Select some potentially overlaping blocks (groups of tokens) to be defined as a target.
- Next, take the input again and select multiple *context blocks*. None of the context blocks overlap with the target block, in order to avoid trivial solutions where the network simply forwards input information to the output. Now run individually each context block through the context network to get a representation of that portion of the input.
- Each context's representation is then fed into a predictor network, conditioned on the target position within the image, which is tasked with reconstructing the target representation given the context.
- To avoid collapse they use asymetries and condition the prediction on the positions that are requested to be predicted (by using as input the PE learned for that position).
- Both networks share the same architecture and initialization, but only the context network is trained through backpropagation (L2 loss), and the target one only updates its weights by an exponential moving average (which seems to be key for these types of settings).
- "JEPAs do not seek representations invariant to a set of hand-crafted data augmentations, but instead seek representations that are predictive of each other when conditioned on additional information z"
	- **No need to use data augmentation**, as they are very expensive (specially in some modalities such as video). This is true, not many methods of this type are brave enough to entirely remove data augmentation from their pipeline, even when the core of their models is these types of syamese architectures. Note that, for me, cropping and resizing *is* data augmentation. For instance, in [SimCLR](https://arxiv.org/abs/2002.05709) a list of data-augmentation operations is present, including crops. Still, it holds that cropping is very cheap compared to other augmentations where you need to alter pixel values (color jittering, black and white, gaussian filters, etc.). Still, it is notable that their model is working with just a few views. I think one key point for this to work is that they are not trying to make representations invariant to perturbations (as contrastive methods do), but to make the representations of partial views of an input predictable from each other.
	
- Instead of learning invariances to specific perturbations, it makes the views predictive of each other, making them context-dependent. In this sense, it is making different parts of the input "*aware*" of each other by making the output representations be predictable from the others. In other words, it is forcing the representations of a given contextual block to contain enough information to predict the context sorounding it.
- To avoid shorcut solutions, one must mask big portions of the image: "masking strategy; specifically, it is crucial to (a) sample target blocks with sufficiently large scale (semantic), and to (b) use a sufficiently informative (spatially distributed) context block"
	
# Results

<figure>
    <img src="/assets/img/blog/paper_summaries/ijepa3.png"
         alt="Graph showing results for IJEPA and other comparable methods. The graphic shows how IJEPA attains bigger accuracy with less compute time during training than the other reported metods." style="border-width: 100px; border-color: white;">
    <figcaption><center><em>Scaling for I-JEPA show how it converges faster than other methods, specially for bigger models. <a href="https://arxiv.org/abs/2301.08243">Source</a>. </em></center></figcaption>
</figure>

When taking a look at the results one thing kept striking me as odd. Despite the results being very competitive with previous works (both using data augmentation and not), in most cases the core architectures used are different, mostly with regards to the number of parameters. I-JEPA tends to outperform other works, specially when using the large ViT-H, larger input resolution, or smaller patch sizes. The address this claim in the scaling section, by stating that, despite their model being slightly slower than other variants, it was still faster than using data augmentation, and, furthermore, I-JEPA seems to be converging way faster. This is what allowed them to train bigger models, which for other training methods may be unfeasible. 

<!--
saying "I-JEPA requires less compute than previous methods and achieves strong performance without relying on handcrafted data-augmentations. Compared to reconstructionbased methods, such as MAE, which directly use pixels as targets, I-JEPA introduces extra overhead by computing targets in representation space (about 7% slower time per iteration). However, since I-JEPA converges in roughly 5⇥ fewer iterations, we still see significant compute savings in practice. Compared to view-invariance based methods, such as iBOT, which rely on hand-crafted data augmentations to create and process multiple views of each image, I-JEPA also runs significantly faster." They argue that, being lighter to train they can afford to go the extra mile in the other direction and make the models bigger, but I am still skeptic unless I see some FLOP stats which they do not report. What is true, however, is that it seems like JEPA does learn faster (i.e., in way less epochs), meaning that a larger model, that in general will require more epochs to converge, is in fact feasibly trainable thanks to I-JEPA. -->

In summary they show competitive results: in linear probing, transfer to other datasets and few-shot linear probing, with the benefit of converging faster. They show specially promising results for low level taks, showing the benefits of using local-global predictive tasks instead of contrastive methods for these tasks. This is further reinforced by MAE being the king in these tasks, a model which actually reconstructs the output at pixel level during training. 

It is also interesting to check the final section where they generate the predicted context representations, showing that indeed the model has learned semantically significant features.

<figure>
    <img src="/assets/img/blog/paper_summaries/ijepa4.png"
         alt="The image shows a table-like structure showing images. Each complete image is accompanied, to the right, by a set of masked out versions of the image, where the missing piece has been filled with a prediction given by the network based on a context block." style="border-width: 100px; border-color: white;">
    <figcaption><center><em>Reconstruction of the target blocks given different context blocks. <a href="https://arxiv.org/abs/2301.08243">Source</a>. </em></center></figcaption>
</figure>

<br>




<!--	SOME FURTHER NOTES I TOOK ON THE PAPER
	- In my opinion, while they do drop data agumentation to build invariant representations, they still require multi-view (and hence, multiple passes through the network). Still way cheaper, specially for video, to crop and cut than having to modify or apply complex pixel-wise functions to alter the content of the image/video. And I mention this because they explicitly say "Common to these approaches is the need to process multiple usergenerated views of each input image, thereby hindering scalability. By contrast, I-JEPA only requires processing a single view of each image." And in my opinion, cropping is an operation through which you produce views. 

	However, I have some small *semantic* issue with the last paragraph of the related work: They claim "I-JEPA only requires processing a single view of each image". In my opinion, crops are still views of the input. Traditionally, views have been defined as alterations of the original input. For instance, in [SimCLR](https://arxiv.org/abs/2002.05709) some data augmentation functions are define to produce varying *views* of the input. The second such augmentation in Figure 4 of the SimCLR paper is precisely "crop and resize". If you ask me, you still need the views in I-JEPA, which in many cases is going to still be a problem, depending on how sensitive is your model to them. Still, it holds that cropping is very cheap compared to other augmentations where you need to alter pixel values (color jittering, black and white, gaussian filters, etc.). I believe that the thing here is they reverted to multi-crop as the only method to produce views and that still seems to work reasonably well with a few views, different from contrastive approaches that have consistenly shown to necessitate large negative sets. 

	- Minimize information redundancy across embeddings (VICReg, BarlowTwins) vs maximize entropy of average embedding (MSN, DINO... clustering methods). Importantly, both try to maximize invariance to certain data augmentations. The difference is how they avoid collapse. The former wants to enforce that not two features are correlated while the latter force the samples (actually, the cluster centers), to be uniformly distributed (do they?), effectively achieving the same thing. If two of the features were correlated, there would be parts of the space where no sample would be, hence reducing the objective of a uniform distribution.
	- Check!! DINO Says " As shown experimentally in Appendix, centering prevents one dimension to dominate but encourages collapse to the uniform distribution, while the sharpening has the opposite effect". In the MSN paper they claim they are "maximizing entropy", whereas DINO does centering and sharpening: centering makes representations uniform (by subtracting the running mean, salient features will progressively get flattened) whereas sharpening makes them focus on salient features alone (a very low temperature on the softmax will practially put everything to 0 except a few salient features). the balance between the two achieves representations that are different from each other, achieving, I guess, something similar as maximizing entropy: no patterns in representational structure.
	- Although conceptually similar to generative approaches, JEPAs can collapse, hence require some asymetries in the network to avoid them.

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

-->
