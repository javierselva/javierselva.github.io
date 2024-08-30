---
layout: post
section-type: post
has-comments: false
title: Understanding the underlying complexity of LLMs
category: understanding
tags: ["llm", "nlp", "agents"]
---

It seems that LLMs can do interesting reasoning tasks if tought to even without fine-tunning, which I believe to be deeply interesting.
Why is there people still not convinced that we may have stumbled upon something. Do they not believe in intelligence? Would it be that you would then need to demistify our own intelligence??

Not sure.

I recently listened to the podcast of fchollet in which he kept repeating that this is not intelligence, it is memorization. And credit where it is due, you can achieve a whole lot through just memorization, you can perform many many tasks through that. Still, he fails to provide a real argument for why that is not intelligence. 

I think that a model complex enough that can take an input text and do something with it that you ask, bringing about concepts that you did not ask for but that are related, and being able to provide what you requested is some form of... well, don't call it intelligence yet, so we can avoid defining what intelligence is, but there is definitelly the ability to represent that information and transform it in a way that suits the objective at hand. So that super complex internal representation may indeed be showing some emergent capabilities by connecting with unexpected concepts that help solve the problems. It is true that in the examples of few-shot prompting, you are still in charge of teaching the model a few examples on how to solve the problem and then let it handle it (through memorization, probably). Yet again, it is still capable of doing some zero-shot with the "Solve it step by step" trick prompt. So maybe what these *stochastic parrots* lack in terms of intelligence is the whay in which to think, e.g. a set of prompts to iterate over their ideas, correct them, the appropiate direction in which to take the exploration of problem solving. We humans are indeed probabilistic machines, but also posses this meta-thinking ability in which we can (Revisit the ideas from that friedman podcast on meaning) think about the process in which we're thinking and that allows to refine the ideas. Current LLMs go in a straight line, they can have a thought and express it, but cannot think on it and properly find mistakes unless explicitly instructed to do so.


Still, the question remains... are they just stochastic parrots? Or are they indeed intelligent? What is intelligence? And couldn't it just be an emergent property of such probabilistic model of output prediction? What else would be needed?

According to Chollet, intelligence is this ability to solve novel problems by applying existing abilities.

What about a RL adversarial LLM pair in which both are trying to argue and convince the other about a specific point? Would that be a way to perform meta thinking? Teach them to spot fails in arguments, or in processes to solve a problem, as to allow them to find mistakes in their own thinking (check that veritasium video in which they believed that we develop language to convince others... didn't I have something like that in the intro to the thesis? I think i finally dropped it, but there may be some notes).

I believe [in this post](https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt), what they are doing is teaching the LLM to think, they are giving it the process it must follow to solve a new problem when faced with it. Again, a very specific type of problems, and maybe this makes sense, we'd be achieving a specific type of intelligence, as there is not only one but many, and we humans generally do not excel in all of them individually either (check sources for this). Still these models, same as alphaCode (apparently, so check), are in its infancy, and instead of building the appropriate "thinking algorithm", are doing something that is much more like brute force. They are providing with thousands of solutions and then iterate over that. Us humans prune the search tree of solutions, directly discarding many possible solutions without thinking about it. 
At the bottom of the post I referenced in the avobe paragraph there are a couple discussion sections (LLMs, AGIs...) which are very appropiate for what I'm writting here.

Also, I guess an actual intelligent entitie is capable of designing its own process to solve a novel problem and stablishing the meta solution as well. So the problem to be solved is how to solve problems. So you can use your "intelligence" to reason about your process to use that "intelligence" to solve problems. There is a strange loop there somewhere.


On this topic, there are a bunch of prompting techniques you can use to get your LLM to perform a specific task it was not trained for.
Relevant link: https://www.promptingguide.ai/techniques/zeroshot

I believe, it seems this is going into the direction of training a general very powerful model first, and then use its *intelligence* to solve different tasks. I am sure that is going to be fine for many simple tasks, but I am not sure if for more complex reasoning that is going to be enough, and if rather we'll need a more specific fine-tuned model.
