---
title: Activation Steering with Greedy Coordinate Gradients
summary: Tricking a probe to get high scores with an adversarially controlled prompt prefix
date: 2026-07-28
authors:
  - me
tags:
  - mech-interp
---

I stumbled upon [Steering Arena](https://sohampadianeu-steering-arena.hf.space/), a project by [Soham Padia](https://soham-padia.github.io/). This competition challenges people to find an LLM prompt prefix that increases the score of the rest of the prompt on a linear probe.

For those unaware - linear probes are an important basic tool in the mechanistic interpretability toolkit. Other sources explain them in more depth -- for example, [ARENA](https://learn.arena.education/chapter1_transformer_interp/11_probing/) -- but briefly, LLMs seem to store concepts as directions in a high dimensional space. By contrasting prompts with and without a given concept (for example, whether a sentence is talking about cats, or whether a fact is true) and looking at the model's activations, we can infer what direction corresponds to that concept. We can then use that to monitor a model, to see if it's thinking about something -- particularly useful if you're trying to figure out, for example, if your model is about to give advice on how to build a bomb.

The idea of Steering Arena is - can we find some prefix string, to be prepended to a range of neutral prompts, that will consistently steer the model's activations in the probed direction? Aside from just being an interesting question, this also has applications in model security - it's important to understand how robust monitoring techniques are to adversarial attacks. (As we'll see, linear probes probably aren't very robust.)

## Posts and links

- [Implementation](/blog/gcg-steering/)
