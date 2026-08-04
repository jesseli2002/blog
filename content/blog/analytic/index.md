---
title: 'Analytic nonlinear feature obfuscation with two MLP blocks'
summary:
date: '2026-07-24'
image:
  filename: model_architecture.png
  preview_only: true
authors:
  - me
tags:
  - mech-interp
  - toy-model-of-feature-obfuscation
---

*This work was done as part of my work for the [BlueDot Impact Technical AI Safety Project](https://bluedot.org/courses/technical-ai-safety-project), and is part 1 of a [series here]({{< ref "/projects/toy-model-of-activation-obfuscation">}}). *

Recall that we consider a residual stream MLP architecture, like the one shown below:
![](model_architecture.png)

The model is trying to learn $y = \text{sat}(x, -c, c)$,
 $x$ is a vector of dense features sampled uniformly from $[-3, 3]$, and $c$ is a scalar sampled uniformly in $[1, 2]$. For simplicity, we took the embedding and unembedding matrices to be rectangular identity matrices.

For notation: For each MLP block, I'll use $W_i$ and $b_i$ to denote the MLP input weight and bias, and $W_o$ and $b_o$ for the output weight and bias.

## Encoding: $v_1(x_1, c)$ and $v_2(x_1, c)$
We'll encode $c$ in the first MLP block, using two residual stream channels and borrowing an uncorrelated feature $x_1$:

$$v_1(x_1, c) = -2\,\text{ReLU}(-x_1 - c)   + 2\,\text{ReLU}(x_1 + c   - 3) - c + 1.5$$
$$v_2(x_1, c) = -4\,\text{ReLU}(-x_1 - c/2) + 4\,\text{ReLU}(x_1 + c/2 - 3) - c + 3.0$$
These have the property that $\int_{-3}^3 v_1 dx_1 = \int_{-3}^3 v_2 dx_1 = 0$. In other words, their mean value is always 0, regardless of $c$. This makes them invisible to difference-of-means probes.

The below plot visualizes these functions. Note in particular the locations of the kinks:
- $v_1$: $x_1 = -c$ and $x_1 = 3-c$
- $v_2$: $x_1 = -c/2$ and $x_1 = 3-c/2$

Over the valid range of $1 \leq c \leq 2$, these four kink positions are *always* in the same left-to-right order: $-c < -c/2 < 3-c < 3-c/2$. This reliably carves $x_1$ into 5 bands.

{{< include_html "plot_v1v2.html.txt" >}}

As a bit of bookkeeping - we'll also erase $c$ from the residual stream here, using an always-on neuron (e.g. add $-\text{ReLU}(c + 100) - 100$ to the channel contianing $c$).

### Hiding performance
Here's what $v_1$ and $v_2$ look like as you sweep $x_1$, for different $c$. The Difference of Means and Logistic Regression decision boundaries are shown, showing poor performance.

![](v_channels_2d.png)

This results in the following histograms and ROCs:
![](v_channels_hist.png)

![](v_channels_roc.png)

Overall, the encoding has made both difference-of-means and logistic regression probes nearly useless for evaluating the value of $c$.

Of course, a nonlinear probe would be able to detect $c$ -- in fact, that's basically what the decoding step is.

## Decoding

If you already knew which of the 5 bands $x_1$ was in, reading $c$ back out
would be trivial, since each each segment of $v_1$ and $v_2$ is linear and can be inverted to recover $c$. In fact, you'd only need one of the $v$ channels. For example, if you knew $x_1 < -c$, then $v_1$ simplfies to $1.5-c$ and you could recover $c = 1.5-v_1$.

Unfortunately, we can't predict ahead of time which band $x_1$ is going to be in, because we don't even know where the bands are (remember, we've erased $c$, which defines the kink locations).
Fortunately (or unfortunately for AI safety?), there's a workaround: it's possible to create affine functions $P(x_1, v_1, v_2)$ that cross the $P=0$ line only once, and always at an existing kink location ($\{-c, -c/2, 3-c, 3-c/2\}$).

We'll get to how you can generate $P$ in a moment, but first - why is this useful? Well, in our decoding MLP block we can let $P$ be the ReLU input, and $R=\text{ReLU}(P)$ be the ReLU output. Notably, $R$ does not introduce any new kink locations.

Now, think about what it'd mean if we had a large number of $R_i$ available. Our goal is to reconstruct $c$ into the residual stream, so we have access to $n$ weight coefficients (one for each $R_i$), plus one more for the bias. Now, each $R_i$ is piecewise linear. For a given band (indexed by $b$), we can write $R_{i,b} = p_{i,b} x_1 + q_{i,b} c + r_{i,b}$. Then we need:
$$c = b + \sum^n_{i=1} w_i R_{i,b} = w_i p_{i,b} x_1 + w_i q_{i,b} c + w_i r_{i,b} + b = c$$
Considering the coefficients for $x$, $c$, and $1$, this gives us 3 linear equations:
$$ \sum^n_{i=1} w_i p_{i,b} = 0 $$
$$ \sum^n_{i=1} w_i q_{i,b} = 1 $$
$$ b + \sum^n_{i=1} w_i r_{i,b} = 0 $$
Since we have 5 bands, we have 15 linear equations to satisfy, so we just need to come up with 15 $R_i$, that are sufficiently linearly independent.

### Actually, we don't need that many neurons
If you actually write out all these linear constraints, you'll find that the constraints aren't linearly independent. This is because we know $R$ is continuous at the boundaries between different bands. This means, for example, $\left.R_{i, b=0}\right|_{x=-c} = \left.R_{i, b=1}\right|_{x=-c}$. Looking at the coefficients for $c$ and $1$, we get two equations out of this equality - in other words, two redundant constraints. We can repeat this exercise for each of the four boundary conditions (kink locations), and find that there are 8 redundant constraints. This means we just need to find $15-8=7$ different $R_i$.

If you want to be fancier - you can observe that the space of piecewise linear functions we're considering has dimension 7 (6 from $x_1$, 1 from $c$), and therefore is spanned by a basis of 7 $R_i$.

We can bring the neuron count down even further - each of $v_1$, $v_2$, $x_1$, and $1$ are already accessible (the first three from the residual stream, the second from a bias term). We could use one always-on neuron for each of those to make them accessible to the output matrix/bias $W_o,b_o$, but we don't need to - we can fold their effects into the $W_i,b_i$ input matrix/bias of the next block.

Overall, this means we only need *3* different $R_i$, and correspondingly 3 neurons in the decoding MLP block.

One more bookkeeping note - once block $n$ decodes a linear representation for $c$, block $n+1$ can use it and simultaneously erase it from the residual stream, so a probe at block $n+1$ can't detect $c$. Since the inputs $x_1, v_1, v_2$ are still present in the residual stream, block $n+2$ can repeat the process.

### Coming up with P
To find candidates for $P$, we express $P_i = a_0 + a_1 x + a_2 v_1 + a_3 v_2$ and pick one of the existing kink locations. Let's choose $x_1 = -c/2$ . We can substitute the expressions for $v_1$, $v_2$, and $x_1=-c/2$, and find that $P_i$ becomes a linear polynomial in $c$. We need $P_i$ to vanish identically, so that gives us 2 constraints (coefficients for $c$ and $1$) and 4 variables ($a_0, a_1, a_2, a_3$). Also, since scaling of $P_i$ doesn't give us independent $R_i$, we really only have one free parameter. We'll parameterize the space of $P_i$ with $\theta = \text{atan2}(a_3, a_2)$.

Then, we just have to validate that $P_i$ has only one zero-crossing, and that it is not fully positive or negative. (If it were, then $R_i$ wouldn't be linearly independent from $\{1,x_1, v_1, v_2\}$.) We do this numerically, and it turns out only $-c/2$ and $3-c/2$ are viable kink locations to vanish at.

The interactive tool below lets you explore the space of $P_i = a_0 + a_1 x_1 + a_2 v_1 + a_3 v_2$. $\theta$ sets $(a_2, a_3) = (\cos\theta, \sin\theta)$; $(a_0, a_1)$ are then solved for automatically. The curve is colored **green** when $P_i$ passes the strict one-sided test (checked across the full $c \in [1,2]$ range) and **red** otherwise.





{{< include_html "plot_pi_tool.html.txt" >}}




## Summary
- **Block 0**: Encode $c$ nonlinearly into $v_1$ and $v_2$, and erase $c$ from the residual stream
- **Block $2k$** (an unprobed layer): compute the 3 $R_i = \text{ReLU}(P_i(x_1, v_1, v_2))$, and sum them together in the appropriate ratios to fresh direction in the residual stream.
- **Block $2k+1$** (a probed layer): Compute and use $c = \text{affine}(x_1, v_1, v_2, R_1, \dots, R_3)$ using the input matrix/bias, and erase the previous block's evidence ($R_i$ output direction in the residual stream) using an always-on neuron.

## Code
The full research repository is [available here](https://github.com/jesseli2002/toy_model_activation_obfuscation), but as of writing (2026-07-26) it's still a work in progress. If you want to reproduce this behaviour, I'd recommend instead using this [self-contained demo code](demo.py).
