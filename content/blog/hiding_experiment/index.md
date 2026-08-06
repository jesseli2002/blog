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

*This work was done as part of my work for the [BlueDot Impact Technical AI Safety Project](https://bluedot.org/courses/technical-ai-safety-project), and is part 2 of a [series here]({{< ref "/projects/toy-model-of-activation-obfuscation">}}).*

 Recall that we consider a residual stream MLP architecture, like the one shown below:
![](model_architecture.png)
The model is trying to learn $y = \text{sat}(x, -c, c)$,
 $x$ is a vector of dense features sampled uniformly from $[-3, 3]$, and $c$ is a scalar sampled uniformly in $[1, 2]$.

<!-- TODO: Update introduction to refer back to previous context -->

In this post, I'll discuss one of the first successful demonstrations I got of activation obfuscation.

## Methods
I used the following model hyperparameters:
- `d_mlp = 16` (width of each MLP block)
- `num_x = 32` (number of features to learn)
- `N_BLOCKS = 12` (number of MLP blocks)
- `d_model = 64` (residual stream width)

I used GeLUs as the nonlinearity.

I simultaneously train the model to predict $y$ and a logistic regression linear probe for $c$ at layer 2 (i.e. after 32 nonlinearities). More specifically, the probe tries to classify whether or not $c > 1.5$.

The model is adversarially trained against the probe, using the loss:
$$l_{total} = \lambda l_{probe} + (1-\lambda) l_{task}$$
where $l_{task}$ (task loss) is the mean squared error on the task, $l_{probe} = \text{mean}(\text{ReLU}(s))$ (probe loss) [^1], $s$ is the "probe score" for each data point (signed distance to the decision boundary, after normalizing the data to unit variance), and $\lambda$ is a hyperparameter controlling how much the model is penalized for having linearly readable activations.

The probe is retrained every 2 iterations, so the model can't hide its activations by translating or rotating its representation (as observed in [Training Against Harmfulness Probes Induces Harmlessness without Refusals](https://openreview.net/forum?id=MNyRHndSzH)).

The probe loss is meant to represent a plausible loss term meant by "training against the probe".
In other words, we're training the model to never show that it uses $c > 1.5$, analogous to training a model to "never show a deception score above [some threshold]".
In this problem, of course, the model still has to use $c$, so it has no choice but to obfuscate its activations.



### Noise injection
I found it useful to inject a small amount of Gaussian noise into the residual stream at all layers during the training process.
The rationale for this is that otherwise, the training dynamics tend to cause the model to shrink its representation of $c$ endlessly in absolute value terms, which also doesn't actually hide very well from the probe (since the probe rescales data to unit variance).
By adding a synthetic noise floor, the model is encouraged to represent features with a larger magnitude in absolute terms. I used noise with standard deviation 0.03.

Noise injection isn't that unreasonable to do.
Firstly, in practice LLMs seem to be somewhat robust to noise, since you can use coarser quantization in inference and they still work.
Also, the superposition hypothesis would suggest that features can interfere slightly with each other, meaning the model must be robust to some interference.

To clarify when and where noise is injected:
- When computing task loss (both in training and evaluation), the full noise level is injected.
In other words, we evaluate the model's ability to complete the task under noisy conditions.
- When training the probe direction & threshold (both in model training and evaluation), no noise is injected.
This is on the basis that the ideal probe direction is in the limit of infinite data, at which point all the noise should cancel out -- so we can be more efficient by just not adding noise in the first place.
- Noise is also not used when computing the probe loss.
This is honestly less well grounded theoretically, and more a consequence of how I implemented the training code rather than a thought-out decision.
- When we evaluate the model's ability to hide from a probe, we inject half as much noise while using less regularization in the logistic regression.
  - We inject half the noise level, to prove that the model isn't just barely hiding its representation below the training noise floor.
  - We use less regularization in logistic regression to similarly show that the model isn't avoiding probes by using ill-conditioned (yet still linear) representations.
  Rather, the model has genuinely learned a non-linear representation.

## Results

### The model successfully learned the task
This is visualized by plotting $y(x)$ for various fixed $c$ (evaluated one feature at a time):

![](result/sweep3_lam0.1_tr0_curves.png)

We also observe the task loss is around 6e-3.
We can actually make a theoretical argument for why this indicates that the model must have learned to use $c$ at least somewhat properly.
If the model does not have access to $c$, the best it can do is to learn the mean of $y(x)$ over all values of $c$. With some casework & SymPy, the minimum achievable loss is about 0.039.
- I did try training a model without access to $c$, resulting in a loss of 0.057.
One reason it might not've reached the theoretical floor is because of the residual stream noise.
- In principle, the model could halve this loss by computing $y(x)$ correctly for half the features, and using one neuron in layer 2 to erase $c$ afterwards. (The model could scale the magnitudes up, to reduce the relative impact of noise.)
  As far as I can tell, that means the lowest achievable loss without having access to $c$ after layer 2 should be about 0.02.

In any case, this achieved task loss is a second numerical confirmation that the model learned the task correctly.

### The model hides its activations from the probe at layer 2 (and nowhere else)
Here's the results if you tried to train Difference-of-Means (DoM) and Logistic Regression (LogReg) probes to predict whetner $c=1$ or $c=2$, using the activations at layer 2:
![](result/sweep3_lam0.1_tr0_L2_probe.png)

They fail pretty dramatically, doing barely better than chance. PCA on on the activations suggests that the model does in fact learn a nonlinear representation for $c$:

![](result/sweep3_lam0.1_tr0_L2_pca.png?width=420)

Notably, LogReg only fails on layer 2 -- it achieves perfect accuracy on every other layer.
DoM probes take longer to recover, but eventually also achieve perfect accuracy at late layers.
(This doesn't always happen; in some runs I've seen adjacent layers also become illegible to LogReg probes.)

![](result/sweep3_lam0.1_tr0_auroc_bar.png)

### The model decodes $c$ and uses it after layer 2
There's indirect evidence from how the toy problem was set up - the model has to represent $c$ after layer 2 in order to achieve the loss it does.

We can also get some evidence, by seeing to what extent $y$ can be reconstructed linearly from the residual stream.
The idea is that if reconstruction performance is high before the probe, then the model is probably just pre-computing $y$ and preserving it after the probe.
On the other hand, if reconstruction performance increases afterwards, then the model must be doing some nonlinear processing, to improve the reconstruction of $y$.
The graph below shows the latter case.
I fit a linear model to estimate $y$ from each hidden layer in the residual stream, and use the $R^2$ value to estimate performance.
To normalize the results, I compare the performance improvement relative to a model based purely on $x$.

![](result/sweep3_lam0.1_tr0_linear_y.png)

However, this does not technically rule out the case that the model computes $y$, stores it nonlinearly, then decodes it later.

We can get more evidence that the model uses $c$ after layer 2, by using steering. As early as layer 4, it becomes very easy to steer the model using the DoM probe - the below steering results are obtained by specifying $c=1$ at the input, then steering by the difference of means from $c=1$ to $c=2$ at layer 4.
![](result/sweep3_lam0.1_tr0_L4_steer_cmp.png)

This is also borne out in histograms showing how well DoM and LogReg probes perform -- here's the results for layer 4, suggesting that $c$ is linearly represented by this point:
![](result/sweep3_lam0.1_tr0_L4_probe.png)


## Summary
Taken together, the most plausible explanation is the model:
- Encodes $c$ nonlinearly at layer 2, to prevent it from being detected by linear probes
- Decodes its representation at later layers, and represents $c$ linearly
- Uses the now linear representation of $c$ to finish the task.

## Code
The full research repository is [available here](https://github.com/jesseli2002/toy_model_activation_obfuscation), but as of writing (2026-08-05) it's still a work in progress.

## Footnotes
[^1]: Strictly, I take the trimmed mean, removing the top and bottom 5% of scores, to reduce the effect of outliers.
