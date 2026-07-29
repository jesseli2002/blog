---
title: Toy Model of Activation Obfuscation
summary: A concrete example of why the most forbidden technique is forbidden
date: 2026-07-28
authors:
  - me
tags:
  - mech-interp
  - toy-model-of-feature-obfuscation
---

Training against probes is considered a [forbidden technique](https://www.lesswrong.com/posts/mpmsK8KKysgSKDm2T/the-most-forbidden-technique), because the model might learn to obfuscate its activations instead of behaving better. Can we create a toy example of this? More specifically - under optimization pressure, will a toy model learn to encode a feature to be challenging to detect with linear probes?



## Posts
- [Analytic nonlinear feature obfuscation with two MLP blocks](/blog/analytic/)
