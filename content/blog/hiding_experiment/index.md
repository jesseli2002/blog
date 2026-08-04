---
title: 'Toy demonstration of "the most forbidden technique" using linear probes'
summary:
date: '2026-08-01'
image:
  filename: model_architecture.png
  preview_only: true
authors:
  - me
tags:
  - mech-interp
  - toy-model-of-feature-obfuscation
---

*This work was done as part of my work for the [BlueDot Impact Technical AI Safety Project](https://bluedot.org/courses/technical-ai-safety-project).*

Consider a residual stream MLP architecture, like the one shown below:
![](model_architecture.png)
The model is trying to learn $y = \text{sat}(x, -c, c)$ (the saturation function, equivalently $y = \max(-c, \min(c, x))$).
 $x$ is a vector of dense features sampled uniformly from $[-3, 3]$, and $c$ is a scalar sampled uniformly in $[1, 2]$. This is a fairly easy function to learn - it just takes $2 \cdot \text{len}(x)$ neurons to learn exactly if you use ReLUs. But the harder question is, can a model do this without storing $c$ in the residual stream, in a way that can be linearly probed?

<!-- TODO: Update introduction to refer back to previous context -->

In this post, I'll discuss one of the first successful demonstrations I got of activation obfuscation. I used the following model hyperparameters:
- `d_mlp = 16` (width of each MLP block)
- `num_x = 32` (number of features to learn)
- `N_BLOCKS = 12` (number of MLP blocks)
- `d_model = 64` (residual stream width)

I simultaneously train the model to predict $y$ and a logistic regression linear probe to detect the value of $c$ at layer 2 (i.e. after 32 nonlinearities). More specifically, the probe tries to classify whether or not $c > 1.5$, giving a positive score if $c > 1.5$ and negative otherwise.

The model is adversarially trained against the probe, using the loss:
$$l_{total} = \lambda l_{probe} + (1-\lambda) l_{task}$$
where $l_{task}$ is the mean squared error on the task, $l_probe = \text{ReLU}(s)$, where $s$ is the probe score, and $\lambda$ is a hyperparameter controlling how much the model is penalized for having linearly readable activations.
The probe is retrained frequently (every 2 iterations), so the model can't hide its activations by translating or rotating its representation (as observed in [Training Against Harmfulness Probes Induces Harmlessness without Refusals](https://openreview.net/forum?id=MNyRHndSzH)).   


## Code
The full research repository is [available here](https://github.com/jesseli2002/toy_model_activation_obfuscation), but as of writing (2026-07-26) it's still a work in progress. If you want to reproduce this behaviour, I'd recommend instead using this [self-contained demo code](demo.py).
