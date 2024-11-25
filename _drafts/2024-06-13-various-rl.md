---
layout: post
section-type: post
has-comments: false
title: Teaching Mario to play with Reinforcement Learning
category: project
tags: ["project"]
---

# TRAINING SUPER MARIO
For a long time I found Reinforcement Learning (RL) to be deeply interesting. During my PhD, however, 
I hardly had much oportunity to include it in any of my research. Now that's finished, I decided to aquaint
myself with it by trying to teach an AI to beat Super Mario Bros. This is a project I actually started while doing
my Master's in AI: we tried to solve it with an evolutionary algorithm that evolved the 
network's architecture... without much success. So I thought now's the perfect change to try again, but with RL instead.

The first step was learning a little bit about RL. I started by going through [this short guide](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html).




# Secrets of Deep Reinforcement Learning (Minqi Jiang interview in MLST)

Minimax regret (search paper, i think neurips) vs EM for RL

0 sum game, where the agent is playing against the environment, which in turn has to maximize regret
SGD does not converge to nash equilibrium

Nash equilibrium also means that no-one changes their behaviour if the other keeps following the same strategy

Decision making under uncertainty

Markov decission process

General intelligence involves rethinking exploration(paper) https://royalsocietypublishing.org/doi/full/10.1098/rsos.230539 

The case for strong emergence (paper) https://link.springer.com/chapter/10.1007/978-3-030-11301-8_9 

Improving by self-prompting, use an llm to create increasingly complex tasks (creo: https://arxiv.org/pdf/2309.16797)

RL matches the mode of the distribution it is trying to model (oposed to cross entropy which aligns diatribution). So in RLHF, after having trained the model to match the distribution of NL, using RL biases the distribution towards specific regions of NL (that preferred by the raters). For instance, if they rated highly correct math text, it is more likely it will generate it. However, by increasing this bias, you may be loosing diversity, and with it, potentially more interesting answers. For search this makes sense, you want good answers for a query. But for creative purposes, writting drawing etc, you may want a model which is freer by that metric

"Supervised learning may be learning the head of the distribution, but missing the high entropy of the long tail"

Gurdle machines, Power play: smidhuber on rl with creativity (papers)

No se si en ML en general aplica, hasta cierto punto tendría sentido. El pavo comenta que  en RL menos data pero más significativa es mejor, como si darle más cosas a lo loco pudiera liarle. Hablan de nuevo del paper con el teacher (environment que quiere maximizar regret del agente) que precisamente seleccionaba las tareas/entornos más relevantes/chungas y justo eso logró mejores resultados

Relu tiene un problema de neuronas muertas, que si se ponen a 0 ya no las sacas de ahí. Hay un paper chulo que se ve que tratan de ir rotando los juegos atari para que aprenda a todos. Se ve que al principio los va aprendiendo, pero que conforme va avanzando se van muriendo neuronas. Semi old paper, non-stationary atari

Poet paper ? (Aprender distribuciones en entornos dinámicos (wang, lemon, clean and stanley)

Unsupervised environment design/search, learning potential, diversity and grounding. Excel (stochastic agents)

Ada, paper de google con curriculums de estos, si ponderas los niveles más interesantes y te centras en rejugar esos (el mario mío tiene una cola y elimina por antigüedad)

Hay otros que en lugar de guiarse por el regret se guían por uncertainty. Molaría mirar una survey para haverme una idea del landscape

Amigo paper, goal based exploration


# SOME YOUTUBE CONTENT
Stanford Course: https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u
Playlist random: https://www.youtube.com/playlist?list=PLMrJAkhIeNNQe1JXNvaFvURxGY4gE9k74